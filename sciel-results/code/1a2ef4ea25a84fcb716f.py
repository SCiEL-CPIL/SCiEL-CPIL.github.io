"""Broadband antireflection coating design; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    ambient = float(problem["ambient_index"])
    substrate = float(problem["substrate_index"])
    nL = float(problem["materials"]["L"])
    nH = float(problem["materials"]["H"])
    lo, hi = problem["band_nm"]
    max_layers = int(problem["max_layers"])
    max_total = float(problem["max_total_nm"])
    max_layer = float(problem["max_layer_nm"])

    # 401 equally spaced wavelengths over [lo, hi]
    wls = np.linspace(lo, hi, 401)
    N = len(wls)
    k = 2.0 * np.pi / wls                       # 1/nm
    kl = k * nL
    kh = k * nH

    # Precompute the per-wavelength phase factors for unit thickness.
    # For a layer of index n and thickness d, phase = k*n*d, so exp(2i*phase)
    # = exp(2i*k*n*d) = (exp(2i*k*n))**d.  We store the base per unit nm.
    # To evaluate a stack fast, we need cos/sin of the accumulated phase per layer.

    t0 = time.time()
    DEADLINE = 115.0   # safety margin under 120s / 150s limits

    # ---- reflectance for a given (labels, thicknesses) ----
    def reflectance(labels, thick):
        m = len(labels)
        # Build 2x2 matrices per wavelength vectorized.
        # We'll compute the product of matrices across layers.
        # Represent each matrix as 4 arrays (m00,m01,m10,m11) of length N.
        # Start: identity
        A00 = np.ones(N); A01 = np.zeros(N)
        A10 = np.zeros(N); A11 = np.ones(N)
        for i in range(m):
            if labels[i] == 'L':
                n = nL; ph = kl * thick[i]
            else:
                n = nH; ph = kh * thick[i]
            c = np.cos(ph); s = np.sin(ph)
            # layer matrix [[c, i*s/n],[i*s*n, c]]
            B00 = c; B01 = 1j * s / n
            B10 = 1j * s * n; B11 = c
            # A = A @ B
            nA00 = A00 * B00 + A01 * B10
            nA01 = A00 * B01 + A01 * B11
            nA10 = A10 * B00 + A11 * B10
            nA11 = A10 * B01 + A11 * B11
            A00, A01, A10, A11 = nA00, nA01, nA10, nA11
        # substrate admittance (normal incidence, lossless)
        Ys = substrate
        Y0 = ambient
        Y = (Ys * A00 + A01) / (Ys * A10 + A11)
        r = (Y0 - Y) / (Y0 + Y)
        R = (r * np.conj(r)).real
        return float(np.mean(R) * 100.0)

    # ---- validity / clamping ----
    def clamp(labels, thick):
        thick = [min(max(float(t), 0.0), max_layer) for t in thick]
        # enforce total thickness budget by trimming from the end if needed
        s = sum(thick)
        if s > max_total:
            excess = s - max_total
            for i in range(len(thick) - 1, -1, -1):
                cut = min(excess, thick[i])
                thick[i] -= cut
                excess -= cut
                if excess <= 1e-9:
                    break
        return thick

    # ---- candidate scoring with caching ----
    def score(labels, thick):
        # trim trailing zero-thickness layers
        m = len(labels)
        while m > 0 and thick[m - 1] <= 1e-9:
            m -= 1
        if m == 0:
            return 4.26
        return reflectance(labels[:m], thick[:m])

    # ---- initial seeds ----
    best = None
    best_val = float('inf')

    def consider(labels, thick):
        nonlocal best, best_val
        thick = clamp(labels, thick)
        v = score(labels, thick)
        if v < best_val:
            best_val = v
            best = (list(labels), list(thick))
        return v

    # 1) quarter-wave stack at band centre
    centre = 0.5 * (lo + hi)
    qw_L = centre / (4.0 * nL)
    qw_H = centre / (4.0 * nH)
    for m in range(1, 13):
        labels = ['L', 'H'] * (m // 2) + (['L'] if m % 2 else [])
        thick = []
        for lab in labels:
            thick.append(qw_L if lab == 'L' else qw_H)
        consider(labels, thick)

    # 2) single quarter-wave L (baseline)
    consider(['L'], [qw_L])

    # 3) random restarts (coarse)
    rng = np.random.default_rng(12345)
    for _ in range(40):
        m = int(rng.integers(2, min(max_layers, 14) + 1))
        labels = [rng.choice(['L', 'H']) for _ in range(m)]
        thick = [float(rng.uniform(0.0, max_layer)) for _ in range(m)]
        consider(labels, thick)

    # ---- local optimization: Nelder-Mead on thicknesses for a fixed label set ----
    from scipy.optimize import minimize

    def optimize_labels(labels, thick0, budget):
        m = len(labels)
        def obj(x):
            x = np.clip(x, 0.0, max_layer)
            s = float(np.sum(x))
            if s > max_total:
                # scale down proportionally (keeps feasibility, cheap)
                x = x * (max_total / s)
            return reflectance(labels, list(x))
        x0 = np.array(thick0, dtype=float)
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-5,
                                    'adaptive': True})
            return list(res.x), float(res.fun)
        except Exception:
            return list(thick0), None

    # ---- hill climbing with simulated-annealing style label mutations ----
    def sa_search(budget):
        nonlocal best, best_val
        # start from current best
        labels = list(best[0]) if best else ['L']
        thick = list(best[1]) if best else [qw_L]
        cur_val = best_val if best else reflectance(labels, thick)
        T = 0.5   # initial temperature (in reflectance-% units)
        cool = 0.97
        it = 0
        while time.time() - t0 < budget:
            it += 1
            # propose a mutation
            r = rng.random()
            if r < 0.6:
                # perturb a random thickness
                if not thick:
                    labels = ['L']; thick = [qw_L]; cur_val = reflectance(labels, thick)
                idx = int(rng.integers(0, len(thick)))
                new_thick = list(thick)
                new_thick[idx] = min(max(thick[idx] + float(rng.normal(0, 40)), 0.0), max_layer)
                cand_labels, cand_thick = labels, new_thick
            elif r < 0.8:
                # add a layer
                if len(labels) < max_layers:
                    idx = int(rng.integers(0, len(labels) + 1))
                    lab = str(rng.choice(['L', 'H']))
                    cand_labels = labels[:idx] + [lab] + labels[idx:]
                    d = qw_L if lab == 'L' else qw_H
                    cand_thick = thick[:idx] + [d] + thick[idx:]
                else:
                    idx = int(rng.integers(0, len(thick)))
                    cand_labels = labels
                    cand_thick = list(thick)
                    cand_thick[idx] = min(max(thick[idx] + float(rng.normal(0, 40)), 0.0), max_layer)
            elif r < 0.9:
                # remove a layer
                if len(labels) > 1:
                    idx = int(rng.integers(0, len(labels)))
                    cand_labels = labels[:idx] + labels[idx + 1:]
                    cand_thick = thick[:idx] + thick[idx + 1:]
                else:
                    idx = int(rng.integers(0, len(thick)))
                    cand_labels = labels
                    cand_thick = list(thick)
                    cand_thick[idx] = min(max(thick[idx] + float(rng.normal(0, 40)), 0.0), max_layer)
            else:
                # flip a layer type
                if labels:
                    idx = int(rng.integers(0, len(labels)))
                    cand_labels = list(labels)
                    cand_labels[idx] = 'H' if labels[idx] == 'L' else 'L'
                    cand_thick = list(thick)
                else:
                    cand_labels = ['L']; cand_thick = [qw_L]

            cand_thick = clamp(cand_labels, cand_thick)
            cand_val = score(cand_labels, cand_thick)
            delta = cand_val - cur_val
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
                labels, thick, cur_val = cand_labels, cand_thick, cand_val
                if cur_val < best_val:
                    best_val = cur_val
                    best = (list(labels), list(thick))
            T *= cool
        return best

    # Phase A: local refine each promising seed
    budget_a = 45.0
    seeds = []
    # collect distinct label patterns from seeds
    seen = set()
    for m in range(1, 11):
        labels = ['L', 'H'] * (m // 2) + (['L'] if m % 2 else [])
        key = tuple(labels)
        if key in seen:
            continue
        seen.add(key)
        thick = [qw_L if l == 'L' else qw_H for l in labels]
        seeds.append((labels, thick))
    # a few random patterns
    for _ in range(12):
        m = int(rng.integers(2, min(max_layers, 12) + 1))
        labels = [str(rng.choice(['L', 'H'])) for _ in range(m)]
        thick = [float(rng.uniform(0.0, max_layer)) for _ in range(m)]
        seeds.append((labels, thick))

    for labels, thick in seeds:
        if time.time() - t0 > budget_a:
            break
        thick_c = clamp(labels, thick)
        ot, ov = optimize_labels(labels, thick_c, budget_a)
        if ov is not None:
            consider(labels, ot)

    # Phase B: simulated annealing from the best found
    sa_search(105.0)

    # Phase C: a final local polish of the best
    if best and time.time() - t0 < 112.0:
        labels, thick = best
        ot, ov = optimize_labels(labels, thick, 112.0)
        if ov is not None:
            consider(labels, ot)

    # Final safety
    if best is None:
        best = (['L'], [qw_L])
    labels, thick = best
    thick = clamp(labels, thick)
    return labels, thick

# EVOLVE-BLOCK-END
