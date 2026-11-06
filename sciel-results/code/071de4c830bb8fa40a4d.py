"""Broadband antireflection coating design via deterministic global DE on a fixed
20-layer alternating L/H stack (L on air, H on glass), vectorized thin-film optics."""
import time
import numpy as np

def _compute_mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Mean normal-incidence reflectance (%) over 401 equally spaced wavelengths,
    computed exactly with the characteristic-matrix method."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    n_wl = len(wavelengths)

    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)

    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    beta = 2.0 * np.pi * n_mat * d  # (n_layers,)

    M00 = np.ones(n_wl, dtype=complex)
    M01 = np.zeros(n_wl, dtype=complex)
    M10 = np.zeros(n_wl, dtype=complex)
    M11 = np.ones(n_wl, dtype=complex)

    for i in range(len(labels)):
        delta = beta[i] / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)

        a = c
        b = 1j * s / n_mat[i]
        cc = 1j * n_mat[i] * s
        dd = c

        nM00 = M00 * a + M01 * cc
        nM01 = M00 * b + M01 * dd
        nM10 = M10 * a + M11 * cc
        nM11 = M10 * b + M11 * dd
        M00, M01, M10, M11 = nM00, nM01, nM10, nM11

    Y = M10 / M00
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2 * 100.0
    return float(np.mean(R))


def design(problem):
    start = time.time()
    DEADLINE = start + 145.0  # 5s safety margin before 150s kill

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    centre = 0.5 * (band_nm[0] + band_nm[1])

    # ------------------------------------------------------------------
    # Fast vectorized objective for a FIXED label sequence.
    # Returns mean R (%) + hard penalty for constraint violation.
    # ------------------------------------------------------------------
    def _make_obj(labels):
        n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
        N = len(labels)
        # Precompute per-layer unit phase so delta = (beta/lam) scales with x.
        # beta_i = 2*pi*n_i*x_i ; delta_i = beta_i/lam = (2*pi*n_i*x_i)/lam
        two_pi = 2.0 * np.pi
        def obj(x):
            x = np.clip(x, 0.0, max_layer_nm)
            beta = two_pi * (n_mat * x)  # (N,)
            M00 = np.ones(401, dtype=complex)
            M01 = np.zeros(401, dtype=complex)
            M10 = np.zeros(401, dtype=complex)
            M11 = np.ones(401, dtype=complex)
            for i in range(N):
                delta = beta[i] / WAVELENGTHS
                c = np.cos(delta)
                s = np.sin(delta)
                b = 1j * s / n_mat[i]
                cc = 1j * n_mat[i] * s
                nM00 = M00 * c + M01 * cc
                nM01 = M00 * b + M01 * c
                nM10 = M10 * c + M11 * cc
                nM11 = M10 * b + M11 * c
                M00, M01, M10, M11 = nM00, nM01, nM10, nM11
            Y = M10 / M00
            r = (n_amb - Y) / (n_amb + Y)
            R = np.abs(r) ** 2 * 100.0
            val = float(np.mean(R))
            total = float(np.sum(x))
            if total > max_total_nm:
                val += 50.0 * (total - max_total_nm) / max_total_nm
            return val
        return obj

    WAVELENGTHS = np.linspace(band_nm[0], band_nm[1], 401)

    # ------------------------------------------------------------------
    # 1) Quick baseline: single L quarter-wave layer.
    # ------------------------------------------------------------------
    base_labels = ["L"]
    base_thick = [centre / (4.0 * n_L)]
    base_thick = [max(0.0, min(t, max_layer_nm)) for t in base_thick]
    best_score = _compute_mean_R(base_labels, base_thick, n_amb, n_sub, n_L, n_H, band_nm)
    best_labels = base_labels[:]
    best_thicknesses = base_thick[:]

    # ------------------------------------------------------------------
    # 2) Global DE on fixed alternating L/H stack of N layers.
    #    L on air (index 1.0) -> L first; H on glass (1.52) -> H last.
    #    So labels = L,H,L,H,... ending in H  =>  N even, start L.
    # ------------------------------------------------------------------
    from scipy.optimize import differential_evolution

    def _run_de(N, popsize, maxiter, tol, time_left):
        if N < 2 or N > max_layers:
            return None, None, float('inf')
        labels = ['L' if i % 2 == 0 else 'H' for i in range(N)]
        # Verify ends: first L, last H (N even).
        if labels[0] != 'L' or labels[-1] != 'H':
            return None, None, float('inf')
        obj = _make_obj(labels)
        # bounds: per-layer [0, max_layer_nm]; total handled by penalty.
        bounds = [(0.0, max_layer_nm) for _ in range(N)]
        x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels])
        # Scale initial to respect total.
        tot = x0.sum()
        if tot > max_total_nm:
            x0 = x0 * (max_total_nm / tot)
        x0 = np.clip(x0, 0.0, max_layer_nm)

        def _callback(xk, convergence):
            return time.time() > (start + time_left)

        try:
            res = differential_evolution(
                obj, bounds,
                x0=x0,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                mutation=(0.4, 1.0),
                recombination=0.9,
                seed=12345,
                init='latinhypercube',
                polish=False,
                callback=_callback,
            )
        except Exception:
            return None, None, float('inf')

        x = np.clip(res.x, 0.0, max_layer_nm)
        total = x.sum()
        if total > max_total_nm:
            x = x * (max_total_nm / total)
            x = np.clip(x, 0.0, max_layer_nm)
        score = _compute_mean_R(labels, list(x), n_amb, n_sub, n_L, n_H, band_nm)
        return labels, list(x), score

    # Try a few stack sizes, largest first (more layers -> better broadband AR),
    # allocating more time to larger N.
    candidates = [20, 18, 16, 14, 12, 10, 8]
    candidates = [N for N in candidates if N <= max_layers]

    for N in candidates:
        if time.time() > DEADLINE - 20.0:
            break
        time_left = max(5.0, DEADLINE - time.time() - 20.0)
        # Bigger N gets more population/iterations.
        if N >= 16:
            popsize, maxiter, tol = 15, 25, 1e-4
        elif N >= 12:
            popsize, maxiter, tol = 12, 20, 1e-4
        else:
            popsize, maxiter, tol = 10, 15, 1e-4
        labels, thick, score = _run_de(N, popsize, maxiter, tol, time_left)
        if labels is not None and score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thick[:]

    # ------------------------------------------------------------------
    # 3) Local polish on the best found (Nelder-Mead via scipy) if time allows.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 12.0 and len(best_labels) >= 2:
        try:
            from scipy.optimize import minimize
            labels = best_labels
            obj = _make_obj(labels)
            x0 = np.array(best_thicknesses)
            x0 = np.clip(x0, 0.0, max_layer_nm)
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            x = np.clip(res.x, 0.0, max_layer_nm)
            total = x.sum()
            if total > max_total_nm:
                x = x * (max_total_nm / total)
                x = np.clip(x, 0.0, max_layer_nm)
            score = _compute_mean_R(labels, list(x), n_amb, n_sub, n_L, n_H, band_nm)
            if score < best_score:
                best_score = score
                best_labels = labels[:]
                best_thicknesses = list(x)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final constraint enforcement.
    # ------------------------------------------------------------------
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]

    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]

    return best_labels, best_thicknesses
