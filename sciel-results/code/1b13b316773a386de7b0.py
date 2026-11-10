import time
import numpy as np
import random


def _reflectance_matrix(matrices, k, d, n):
    """Apply characteristic matrices for layers with indices n, thicknesses d at wavenumbers k.
    Returns the product M = M1 M2 ... Mm for layers in air->substrate order (leftmost first).
    Vectorized: k, d, n are 1-D arrays of length m; result shape (m, 2, 2) complex."""
    c = np.cos(k * d)
    s = np.sin(k * d)
    # M_i = [[c, i*s/n_i], [i*n_i*s, c]]
    M = np.empty((len(n), 2, 2), dtype=complex)
    M[:, 0, 0] = c
    M[:, 0, 1] = 1j * s / n
    M[:, 1, 0] = 1j * n * s
    M[:, 1, 1] = c
    # product M1 @ M2 @ ... @ Mm
    result = M[0].copy()
    for i in range(1, len(n)):
        result = result @ M[i]
    return result


def _mean_reflectance_percent(thicknesses, labels, n0, ns, nmat, wl):
    """Mean reflectance (%) over wavelength array wl for a stack."""
    m = len(thicknesses)
    if m == 0:
        # bare substrate
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * float(np.abs(r) ** 2)
    n = np.array([nmat[l] for l in labels], dtype=float)
    d = np.asarray(thicknesses, dtype=float)
    k = 2.0 * np.pi / wl
    M = _reflectance_matrix(None, k, d, n)
    # M shape (len(wl),2,2)
    M11 = M[:, 0, 0]
    M12 = M[:, 0, 1]
    M21 = M[:, 1, 0]
    M22 = M[:, 1, 1]
    Y = (n0 * M11 + 1j * M12) / (n0 * M21 + 1j * M22)
    r = (1 - Y) / (1 + Y)
    R = np.abs(r) ** 2
    return 100.0 * float(np.mean(R))


def design(problem):
    deadline = time.time() + 45.0  # safety margin well under 60s CPU / 120s wall

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nmat = problem["materials"]
    nL = nmat["L"]
    nH = nmat["H"]
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    wl = np.linspace(wl_min, wl_max, 401)
    lam_c = 0.5 * (wl_min + wl_max)

    # Baseline: bare glass
    base_r = (n0 - ns) / (n0 + ns)
    base_score = 100.0 * float(base_r ** 2)

    # --- Build a good physical starting point: 5-layer quarter-wave L H L H L ---
    # Quarter-wave optical thickness at centre wavelength
    qL = lam_c / (4.0 * nL)
    qH = lam_c / (4.0 * nH)
    start_labels = ["L", "H", "L", "H", "L"]
    start_th = [qL, qH, qL, qH, qL]
    # Scale to keep total under max_total_nm
    tot = sum(start_th)
    if tot > max_total_nm:
        scale = max_total_nm / tot
        start_th = [t * scale for t in start_th]
    # Clip each layer to max_layer_nm
    start_th = [min(max(t, 0.0), max_layer_nm) for t in start_th]

    best_labels = start_labels
    best_th = start_th
    best_score = _mean_reflectance_percent(best_th, best_labels, n0, ns, nmat, wl)

    def evaluate(labels, th):
        # enforce constraints; if violated return inf
        if len(th) != len(labels):
            return float("inf")
        for t in th:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return float("inf")
        thc = [min(max(t, 0.0), max_layer_nm) for t in th]
        if sum(thc) > max_total_nm + 1e-6:
            return float("inf")
        return _mean_reflectance_percent(thc, labels, n0, ns, nmat, wl)

    # Local optimization: Nelder-Mead style via scipy if available, else simple coordinate descent
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except Exception:
        have_scipy = False

    def optimize_local(labels, th0, budget):
        th0 = np.asarray(th0, dtype=float)
        m = len(th0)
        if have_scipy:
            def obj(x):
                if time.time() > budget:
                    return float("inf")
                # soft penalty for constraint violation to keep search bounded
                pen = 0.0
                for t in x:
                    if t < 0:
                        pen += 1e4 * (-t) ** 2
                    if t > max_layer_nm:
                        pen += 1e4 * (t - max_layer_nm) ** 2
                s = float(np.sum(x))
                if s > max_total_nm:
                    pen += 1e4 * (s - max_total_nm) ** 2
                val = _mean_reflectance_percent(np.clip(x, 0, max_layer_nm), labels, n0, ns, nmat, wl)
                return val + pen
            try:
                res = minimize(
                    obj, th0, method="Nelder-Mead",
                    options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-5, "adaptive": True},
                )
                x = np.clip(res.x, 0, max_layer_nm)
                # fix total
                if float(np.sum(x)) > max_total_nm:
                    x = x * (max_total_nm / float(np.sum(x)))
                    x = np.clip(x, 0, max_layer_nm)
                return x
            except Exception:
                return th0
        else:
            # coordinate descent fallback
            x = th0.copy()
            step = 20.0
            for _ in range(400):
                if time.time() > budget:
                    break
                improved = False
                for i in range(m):
                    for delta in (step, -step):
                        x2 = x.copy()
                        x2[i] = x[i] + delta
                        if x2[i] < 0 or x2[i] > max_layer_nm:
                            continue
                        if float(np.sum(x2)) > max_total_nm:
                            continue
                        v2 = _mean_reflectance_percent(x2, labels, n0, ns, nmat, wl)
                        v0 = _mean_reflectance_percent(x, labels, n0, ns, nmat, wl)
                        if v2 < v0:
                            x = x2
                            improved = True
                            break
                if not improved:
                    step *= 0.7
                    if step < 0.5:
                        break
            return x

    # Multi-start local search
    n_starts = 8
    random.seed(12345)
    for s in range(n_starts):
        if time.time() > deadline:
            break
        if s == 0:
            labels = list(best_labels)
            th = list(best_th)
        else:
            # random perturbation of the best, or a fresh random stack
            if random.random() < 0.5:
                labels = list(best_labels)
                th = [max(0.0, min(max_layer_nm, t + random.uniform(-40, 40))) for t in best_th]
            else:
                # random 5-layer stack
                labels = random.choice([
                    ["L", "H", "L", "H", "L"],
                    ["H", "L", "H", "L", "H"],
                    ["L", "H", "L", "H", "L", "H"],
                    ["H", "L", "H", "L", "H", "L"],
                    ["L", "H", "L", "H"],
                    ["H", "L", "H", "L"],
                ])
                th = [random.uniform(50, 200) for _ in labels]
            # normalize total
            tot = float(np.sum(th))
            if tot > max_total_nm:
                th = [t * (max_total_nm / tot) for t in th]
            th = [min(max(t, 0.0), max_layer_nm) for t in th]

        budget = min(deadline, time.time() + 4.0)
        xopt = optimize_local(labels, th, budget)
        sc = evaluate(labels, xopt)
        if sc < best_score:
            best_score = sc
            best_th = [float(t) for t in xopt]
            best_labels = list(labels)

    # Final constraint cleanup
    best_th = [min(max(float(t), 0.0), max_layer_nm) for t in best_th]
    if sum(best_th) > max_total_nm:
        scale = max_total_nm / sum(best_th)
        best_th = [t * scale for t in best_th]
        best_th = [min(max(t, 0.0), max_layer_nm) for t in best_th]

    # Ensure at least one layer
    if len(best_th) == 0:
        return ["L"], [lam_c / (4.0 * nL)]

    return best_labels, best_th
