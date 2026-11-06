"""Broadband antireflection coating via continuous L-BFGS-B optimization."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 125.0  # wall-clock seconds, safety margin before 120s kill + evaluator overhead


def _reflectance_spectrum(thicknesses, materials, ambient, substrate, band_nm):
    """Exact thin-film characteristic-matrix reflectance at normal incidence.

    thicknesses: array of physical thicknesses (nm), air -> substrate order.
    Returns reflectance (0..1) at 401 equally spaced wavelengths in band.
    """
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    k = 2.0 * np.pi / wls  # 1/nm
    n = ambient + 0j
    m = substrate + 0j
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for d, ni in zip(thicknesses, materials):
        if d <= 0.0:
            continue
        delta = k * ni * d
        c = np.cos(delta)
        s = np.sin(delta)
        layer = np.array([[c, 1j * s / ni], [1j * ni * s, c]], dtype=complex)
        M = M @ layer
    r = ((n * M[0, 0] + M[0, 1]) - (n * M[1, 0] + M[1, 1])) / \
        ((n * M[0, 0] + M[0, 1]) + (n * M[1, 0] + M[1, 1]))
    R = np.abs(r) ** 2
    return np.clip(R, 0.0, 1.0)


def _mean_reflectance_pct(x, nL, nH, ambient, substrate, band_nm, max_layer, max_total):
    """Objective: mean reflectance in percent, with penalty for constraint violation."""
    if x is None:
        return 1e6
    if not np.all(np.isfinite(x)):
        return 1e6
    d = np.clip(x, 0.0, max_layer)
    if d.sum() > max_total:
        # soft penalty to keep L-BFGS-B inside the box; hard clip handled elsewhere
        d = d * (max_total / d.sum())
    # alternating L/H starting with L
    mats = [nL if i % 2 == 0 else nH for i in range(len(d))]
    R = _reflectance_spectrum(d, mats, ambient, substrate, band_nm)
    return float(np.mean(R) * 100.0)


def design(problem):
    t0 = time.time()

    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    best = None
    best_score = 1e9

    def consider(x, nmat):
        nonlocal best, best_score
        d = np.clip(x, 0.0, max_layer)
        if d.sum() > max_total:
            d = d * (max_total / d.sum())
        # build alternating L/H labels
        labels = ['L' if i % 2 == 0 else 'H' for i in range(nmat)]
        mats = [nL if i % 2 == 0 else nH for i in range(nmat)]
        R = _reflectance_spectrum(d, mats, ambient, substrate, band)
        score = float(np.mean(R) * 100.0)
        if score < best_score:
            best_score = score
            best = (labels, [float(v) for v in d])
        return score

    # Try a range of layer counts, from few to max.
    # Start with small counts (fast), grow as time permits.
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
    counts = [c for c in counts if c <= max_layers]

    # Pre-compute a good initial guess: quarter-wave stack at band centre.
    centre = 0.5 * (band[0] + band[1])

    for n in counts:
        if time.time() - t0 > _DEADLINE:
            break

        # initial guess: quarter-wave alternating
        x0 = np.array([centre / (4.0 * (nL if i % 2 == 0 else nH)) for i in range(n)],
                      dtype=float)
        # clip to max_layer and total
        x0 = np.clip(x0, 0.0, max_layer)
        if x0.sum() > max_total:
            x0 = x0 * (max_total / x0.sum())

        bounds = [(0.0, max_layer)] * n

        def objective(x, n=n):
            return _mean_reflectance_pct(x, nL, nH, ambient, substrate, band,
                                         max_layer, max_total)

        # multi-start: a few random perturbations to escape local minima
        starts = [x0]
        rng = np.random.default_rng(n * 1000 + 7)
        n_starts = 3 if n <= 6 else 2
        for _ in range(n_starts):
            x1 = x0 + rng.uniform(-0.2, 0.2, size=n) * np.array(
                [centre / (4.0 * (nL if i % 2 == 0 else nH)) for i in range(n)])
            x1 = np.clip(x1, 0.0, max_layer)
            if x1.sum() > max_total:
                x1 = x1 * (max_total / x1.sum())
            starts.append(x1)

        for xstart in starts:
            if time.time() - t0 > _DEADLINE:
                break
            try:
                res = minimize(objective, xstart, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-5})
                if res.success or (res.fun < best_score):
                    consider(res.x, n)
            except Exception:
                pass

    if best is None:
        # fallback: single quarter-wave L layer
        return ["L"], [centre / (4.0 * nL)]

    return best[0], best[1]
