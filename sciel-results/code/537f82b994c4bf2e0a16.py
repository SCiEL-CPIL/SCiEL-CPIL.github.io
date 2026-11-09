"""Antireflection coating design: quarter-wave stack + L-BFGS-B refinement."""
import time
import numpy as np

_DEADLINE_SECONDS = 140.0  # safety margin below the 150 s wall-clock kill


def _reflectance_spectrum(thicknesses, materials, n_amb, n_sub, band_nm):
    """Exact characteristic-matrix reflectance (percent) at 401 wavelengths."""
    wl = np.linspace(band_nm[0], band_nm[1], 401)
    n_layers = len(thicknesses)
    n = np.array([materials[l] for l in _labels])
    d = np.array(thicknesses, dtype=float)
    k = 2.0 * np.pi / wl[:, None]  # (401, n_layers)
    # Per-layer transfer matrix (normal incidence, lossless):
    # M_j = [[cos delta_j, i sin delta_j / n_j], [i n_j sin delta_j, cos delta_j]]
    cos_d = np.cos(k * n[None, :] * d[None, :])
    sin_d = np.sin(k * n[None, :] * d[None, :])
    # Build total matrix product M1 M2 ... Mn
    # Represent each matrix as (401, 2, 2)
    m11 = cos_d
    m12 = 1j * sin_d / n[None, :]
    m21 = 1j * n[None, :] * sin_d
    m22 = cos_d
    # Product via repeated multiplication
    T11 = np.ones_like(wl, dtype=complex)
    T12 = np.zeros_like(wl, dtype=complex)
    T21 = np.zeros_like(wl, dtype=complex)
    T22 = np.ones_like(wl, dtype=complex)
    for j in range(n_layers):
        a11, a12 = m11[:, j], m12[:, j]
        a21, a22 = m21[:, j], m22[:, j]
        nT11 = T11 * a11 + T12 * a21
        nT12 = T11 * a12 + T12 * a22
        nT21 = T21 * a11 + T22 * a21
        nT22 = T21 * a12 + T22 * a22
        T11, T12, T21, T22 = nT11, nT12, nT21, nT22
    # Reflectance: r = (n_amb*T11 + T12*n_sub - T21 - T22*n_amb) / (n_amb*T11 + T12*n_sub + T21 + T22*n_amb)
    num = n_amb * T11 + T12 * n_sub - T21 - T22 * n_amb
    den = n_amb * T11 + T12 * n_sub + T21 + T22 * n_amb
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * R


_labels = []  # set inside design()


def _objective(x, materials, n_amb, n_sub, band_nm, max_layer, max_total):
    """Penalized mean reflectance."""
    x = np.clip(x, 0.0, max_layer)
    if x.sum() > max_total:
        # scale down proportionally to satisfy total constraint
        x = x * (max_total / x.sum())
    global _labels
    R = _reflectance_spectrum(x, materials, n_amb, n_sub, band_nm)
    return float(np.mean(R))


def _quarter_wave_stack(n_layers, materials, band_nm):
    """Build alternating L/H quarter-wave layers (at band center), starting with L."""
    centre = 0.5 * (band_nm[0] + band_nm[1])
    labels = []
    thicks = []
    for i in range(n_layers):
        lab = "L" if i % 2 == 0 else "H"
        labels.append(lab)
        thicks.append(centre / (4.0 * materials[lab]))
    return labels, thicks


def design(problem):
    global _labels
    t_start = time.time()
    materials = problem["materials"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    # Choose a number of layers: use a generous count for broadband coverage.
    # Even count so the stack starts with L (air side, low index ~ matches air better)
    # and ends with H (high index near substrate 1.52).
    n_layers = 18

    labels, thicks0 = _quarter_wave_stack(n_layers, materials, band_nm)
    _labels = labels

    # Enforce initial feasibility
    x0 = np.array(thicks0, dtype=float)
    x0 = np.clip(x0, 0.0, max_layer)
    if x0.sum() > max_total:
        x0 = x0 * (max_total / x0.sum())

    best_x = x0.copy()
    best_val = _objective(best_x, materials, n_amb, n_sub, band_nm, max_layer, max_total)

    # Local refinement with L-BFGS-B
    try:
        from scipy.optimize import minimize
        t_opt_start = time.time()
        res = minimize(
            lambda x: _objective(x, materials, n_amb, n_sub, band_nm, max_layer, max_total),
            x0,
            method="L-BFGS-B",
            bounds=[(0.0, max_layer)] * n_layers,
            options={"maxiter": 400, "maxfun": 20000, "gtol": 1e-5, "maxls": 20},
        )
        if res.x is not None:
            x_opt = np.clip(res.x, 0.0, max_layer)
            if x_opt.sum() > max_total:
                x_opt = x_opt * (max_total / x_opt.sum())
            val_opt = _objective(x_opt, materials, n_amb, n_sub, band_nm, max_layer, max_total)
            if val_opt < best_val:
                best_x = x_opt
                best_val = val_opt
    except Exception:
        pass

    # If time remains, do a few random-restart local refinements to escape local minima.
    while time.time() - t_start < _DEADLINE_SECONDS - 20.0:
        # Perturb best solution
        jitter = np.random.uniform(-0.15, 0.15, size=n_layers)
        x_try = best_x * (1.0 + jitter)
        x_try = np.clip(x_try, 0.0, max_layer)
        if x_try.sum() > max_total:
            x_try = x_try * (max_total / x_try.sum())
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda x: _objective(x, materials, n_amb, n_sub, band_nm, max_layer, max_total),
                x_try,
                method="L-BFGS-B",
                bounds=[(0.0, max_layer)] * n_layers,
                options={"maxiter": 100, "maxfun": 5000, "gtol": 1e-4, "maxls": 15},
            )
            if res.x is not None:
                x_opt = np.clip(res.x, 0.0, max_layer)
                if x_opt.sum() > max_total:
                    x_opt = x_opt * (max_total / x_opt.sum())
                val_opt = _objective(x_opt, materials, n_amb, n_sub, band_nm, max_layer, max_total)
                if val_opt < best_val:
                    best_x = x_opt
                    best_val = val_opt
        except Exception:
            pass
        # Break if we're getting close to the deadline
        if time.time() - t_start > _DEADLINE_SECONDS - 10.0:
            break

    # Final feasibility check
    best_x = np.clip(best_x, 0.0, max_layer)
    if best_x.sum() > max_total:
        best_x = best_x * (max_total / best_x.sum())

    return _labels, [float(v) for v in best_x]
