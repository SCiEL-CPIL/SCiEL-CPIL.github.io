import time
import numpy as np

def _compute_reflectance(wavelengths, n_amb, n_sub, materials, labels, thicknesses):
    """Exact normal-incidence reflectance (fraction) via the characteristic-matrix method."""
    n = len(labels)
    nL = np.array([materials[l] for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    k = 2.0 * np.pi / wavelengths
    # phase per layer (vector over wavelengths)
    phi = k * nL * d  # shape (n, m)
    c = np.cos(phi)
    s = np.sin(phi)
    # Y = n * cos(phi) / (i sin(phi)) ; M = [[cos, i sin/n],[i n sin, cos]]
    # Build 2x2 matrices in complex arrays: shape (n, 2, 2, m)
    M00 = c
    M01 = 1j * s / nL
    M10 = 1j * nL * s
    M11 = c
    # Start with substrate admittance
    Y = np.full(wavelengths.shape, n_sub, dtype=complex)
    # Multiply matrices from last layer to first (air side is index 0)
    for i in range(n - 1, -1, -1):
        M00i = M00[i]; M01i = M01[i]; M10i = M10[i]; M11i = M11[i]
        Y = (M00i * Y + M01i) / (M10i * Y + M11i)
    R = ((Y - n_amb) / (Y + n_amb)) ** 2
    R = R.real
    return R

def design(problem):
    t_start = time.time()
    DEADLINE = t_start + 145.0  # safety margin under 150s

    n_amb = float(problem["ambient_index"])
    n_sub = float(problem["substrate_index"])
    materials = problem["materials"]
    band = problem["band_nm"]
    max_layers = int(problem["max_layers"])
    max_total = float(problem["max_total_nm"])
    max_layer = float(problem["max_layer_nm"])

    wl = np.linspace(band[0], band[1], 401)
    n_eval = len(wl)

    def mean_r(labels, thicknesses):
        # quick constraint check
        if len(labels) > max_layers:
            return 1.0
        if any(t < -1e-6 or t > max_layer + 1e-6 for t in thicknesses):
            return 1.0
        if sum(thicknesses) > max_total + 1e-6:
            return 1.0
        R = _compute_reflectance(wl, n_amb, n_sub, materials, labels, thicknesses)
        return float(np.mean(R))

    def clamp(th):
        return np.clip(th, 0.0, max_layer)

    # candidate layer patterns
    def alt_labels(n, start):
        return [start if i % 2 == 0 else ('H' if start == 'L' else 'L') for i in range(n)]

    patterns = []
    for n in range(2, min(12, max_layers) + 1):
        patterns.append(alt_labels(n, 'L'))
        patterns.append(alt_labels(n, 'H'))
    # a couple of non-alternating patterns
    patterns.append(['L','H','H','L'])
    patterns.append(['H','L','L','H'])
    patterns.append(['L','H','L','H','L'])

    best_labels = None
    best_th = None
    best_val = float('inf')

    def try_stack(labels, th):
        nonlocal best_labels, best_th, best_val
        th = [float(t) for t in th]
        if any(t < -1e-9 or t > max_layer + 1e-9 for t in th):
            return
        if sum(th) > max_total + 1e-9:
            return
        val = mean_r(labels, th)
        if val < best_val:
            best_val = val
            best_labels = list(labels)
            best_th = list(th)

    # initial heuristic: quarter-wave at centre, scaled to fit
    centre = 0.5 * (band[0] + band[1])
    for labels in patterns:
        n = len(labels)
        nL = np.array([materials[l] for l in labels], dtype=float)
        q = centre / (4.0 * nL)
        # scale to fit total budget
        tot = q.sum()
        if tot > max_total:
            q = q * (max_total / tot)
        q = np.clip(q, 0.0, max_layer)
        try_stack(labels, q.tolist())

    # If we have a decent baseline, do simulated annealing on thicknesses per pattern
    from random import random, randint

    rng_seed = 12345
    import random as _rnd
    _rnd.seed(rng_seed)

    def anneal(labels, th0, budget):
        nonlocal best_labels, best_th, best_val
        n = len(labels)
        nL = np.array([materials[l] for l in labels], dtype=float)
        cur = np.array(th0, dtype=float)
        cur = np.clip(cur, 0.0, max_layer)
        cur_val = mean_r(labels, cur.tolist())
        best_local = cur.copy()
        best_local_val = cur_val
        try_stack(labels, cur.tolist())

        # temperature schedule
        T = 0.004
        T_end = 1e-6
        it = 0
        step_scale = 30.0  # nm initial step
        while True:
            if time.time() > budget:
                break
            it += 1
            # pick a random layer to perturb
            idx = _rnd.randrange(n)
            delta = _rnd.gauss(0.0, step_scale)
            new = cur.copy()
            new[idx] = np.clip(cur[idx] + delta, 0.0, max_layer)
            if new.sum() > max_total:
                # scale down proportionally
                new = new * (max_total / new.sum())
                new = np.clip(new, 0.0, max_layer)
            new_val = mean_r(labels, new.tolist())
            dE = new_val - cur_val
            if dE < 0 or _rnd.random() < np.exp(-dE / max(T, 1e-12)):
                cur = new
                cur_val = new_val
                if cur_val < best_local_val:
                    best_local_val = cur_val
                    best_local = cur.copy()
                    try_stack(labels, cur.tolist())
            # cool
            T = max(T * 0.9995, T_end)
            step_scale = max(step_scale * 0.9995, 5.0)

        return best_local, best_local_val

    # Anneal each pattern, spending time proportionally; prioritize more layers
    order = sorted(range(len(patterns)), key=lambda i: -len(patterns[i]))
    for pi in order:
        if time.time() > DEADLINE - 20.0:
            break
        labels = patterns[pi]
        n = len(labels)
        nL = np.array([materials[l] for l in labels], dtype=float)
        q = centre / (4.0 * nL)
        tot = q.sum()
        if tot > max_total:
            q = q * (max_total / tot)
        q = np.clip(q, 0.0, max_layer)
        # give each pattern a time budget
        pat_budget = min(DEADLINE - 15.0, time.time() + 25.0)
        if time.time() > DEADLINE - 15.0:
            break
        anneal(labels, q.tolist(), pat_budget)

    # Final polish with Nelder-Mead on the best stack found
    if best_labels is not None and time.time() < DEADLINE - 8.0:
        from scipy.optimize import minimize
        labels = best_labels
        n = len(labels)
        nL = np.array([materials[l] for l in labels], dtype=float)

        def obj(th):
            th = np.clip(th, 0.0, max_layer)
            if th.sum() > max_total:
                return 1.0
            return mean_r(labels, th.tolist())

        x0 = np.array(best_th, dtype=float)
        # a couple of Nelder-Mead attempts from slightly perturbed starts
        for attempt in range(3):
            if time.time() > DEADLINE - 6.0:
                break
            if attempt == 0:
                x0k = x0.copy()
            else:
                x0k = x0 + _rnd.gauss(0.0, 10.0, size=n)
                x0k = np.clip(x0k, 0.0, max_layer)
            try:
                res = minimize(obj, x0k, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 0.05, 'fatol': 1e-5,
                                        'adaptive': True})
                th_opt = np.clip(res.x, 0.0, max_layer)
                if th_opt.sum() > max_total:
                    th_opt = th_opt * (max_total / th_opt.sum())
                    th_opt = np.clip(th_opt, 0.0, max_layer)
                try_stack(labels, th_opt.tolist())
            except Exception:
                pass

    # Ensure we always return something valid; fall back to a simple quarter-wave stack
    if best_labels is None:
        best_labels = ['L']
        best_th = [centre / (4.0 * materials['L'])]
    # final constraint enforcement
    best_th = [float(min(max(t, 0.0), max_layer)) for t in best_th]
    if sum(best_th) > max_total:
        scale = max_total / sum(best_th)
        best_th = [t * scale for t in best_th]
    if len(best_labels) > max_layers:
        best_labels = best_labels[:max_layers]
        best_th = best_th[:max_layers]
    return best_labels, best_th
