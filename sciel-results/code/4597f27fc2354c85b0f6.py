"""Broadband antireflection coating design via thin-film characteristic matrix."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401):
    """Mean reflectance (%) over n_wl equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    lams = np.linspace(lam_min, lam_max, n_wl)
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)
    beta = 2.0 * np.pi * n_mat * d  # beta_i = 2*pi*n_i*d_i ; delta_i = beta_i/lambda
    total = 0.0
    chunk = 64
    for s in range(0, n_wl, chunk):
        e = min(s + chunk, n_wl)
        L = lams[s:e]
        m = L.shape[0]
        M00 = np.ones(m, dtype=complex); M01 = np.zeros(m, dtype=complex)
        M10 = np.zeros(m, dtype=complex); M11 = np.ones(m, dtype=complex)
        for i in range(len(labels)):
            delta = beta[i] / L
            c = np.cos(delta); sn = np.sin(delta)
            b = 1j * sn / n_mat[i]
            cc = 1j * n_mat[i] * sn
            n00 = M00 * c + M01 * cc
            n01 = M00 * b + M01 * c
            n10 = M10 * c + M11 * cc
            n11 = M10 * b + M11 * c
            M00, M01, M10, M11 = n00, n01, n10, n11
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        total += np.sum(np.abs(r) ** 2 * 100.0)
    return total / n_wl


def _nm_optimize(labels, x0, n_amb, n_sub, n_L, n_H, band_nm, max_layer_nm,
                 max_total_nm, deadline, iters=250, n_wl=401):
    """Constrained Nelder-Mead (projection onto box + total-thickness simplex)."""
    n = len(labels)
    if n == 0:
        return None, float('inf')

    def proj(x):
        x = np.clip(x, 0.0, max_layer_nm)
        tot = x.sum()
        if tot > max_total_nm:
            x = x * (max_total_nm / tot)
        return x

    def obj(x):
        return _mean_R(labels, x, n_amb, n_sub, n_L, n_H, band_nm, n_wl)

    x0 = proj(x0)
    simplex = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] = np.clip(p[i] + 0.12 * max_layer_nm, 0.0, max_layer_nm)
        simplex.append(proj(p))
    fvals = [obj(v) for v in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for it in range(iters):
        if time.time() > deadline - 3.0:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < 1e-9:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = proj(centroid + alpha * (centroid - simplex[-1]))
        fr = obj(xr)
        if fr < fvals[0]:
            xe = proj(centroid + gamma * (xr - centroid))
            fe = obj(xe)
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = proj(centroid + rho * (xr - centroid)) if fr < fvals[-1] \
                else proj(centroid - rho * (simplex[-1] - centroid))
            fc = obj(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = proj(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    fvals[i] = obj(simplex[i])
    bi = int(np.argmin(fvals))
    return simplex[bi].tolist(), fvals[bi]


def _nm_optimize_seq(labels, x0, n_amb, n_sub, n_L, n_H, band_nm, max_layer_nm,
                     max_total_nm, deadline, iters=250, n_wl=401):
    """Optimize thicknesses, then refine the material sequence via swaps."""
    thick, score = _nm_optimize(labels, x0, n_amb, n_sub, n_L, n_H, band_nm,
                                max_layer_nm, max_total_nm, deadline, iters, n_wl)
    if thick is None:
        return None, None, float('inf')
    n = len(labels)
    for _ in range(4):
        if time.time() > deadline - 5.0:
            break
        best = (labels[:], thick[:], score)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                cand = labels[:]
                cand[i], cand[j] = labels[j], labels[i]
                t, s = _nm_optimize(cand, thick, n_amb, n_sub, n_L, n_H, band_nm,
                                    max_layer_nm, max_total_nm, deadline, 80, n_wl)
                if t is not None and s < best[2]:
                    best = (cand, t, s)
        if best[2] >= score - 1e-12:
            break
        labels, thick, score = best
    return labels, thick, score


def design(problem):
    start = time.time()
    deadline = start + 140.0  # 10s safety margin under 150s wall clock

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    centre = 0.5 * (band_nm[0] + band_nm[1])
    best_labels, best_thick, best_score = None, None, float('inf')

    def upd(labels, thick, score):
        nonlocal best_labels, best_thick, best_score
        if score < best_score:
            best_labels, best_thick, best_score = labels[:], thick[:], score

    # --- 1. Fast coarse scan of alternating stacks (cheap wavelength grid) ---
    for n in range(1, min(max_layers + 1, 11)):
        for start_m in ('L', 'H'):
            seq = [start_m if i % 2 == 0 else ('H' if start_m == 'L' else 'L')
                   for i in range(n)]
            x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq])
            if x0.sum() > max_total_nm:
                x0 = x0 * (max_total_nm / x0.sum())
            x0 = np.clip(x0, 0.0, max_layer_nm)
            s = _mean_R(seq, x0, n_amb, n_sub, n_L, n_H, band_nm, n_wl=101)
            if s < best_score:
                best_labels, best_thick, best_score = seq[:], x0.tolist(), s

    # --- 2. Full-resolution local refinement of promising alternating stacks ---
    for n in range(1, min(max_layers + 1, 9)):
        for start_m in ('L', 'H'):
            seq = [start_m if i % 2 == 0 else ('H' if start_m == 'L' else 'L')
                   for i in range(n)]
            x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq])
            if x0.sum() > max_total_nm:
                x0 = x0 * (max_total_nm / x0.sum())
            x0 = np.clip(x0, 0.0, max_layer_nm)
            thick, score = _nm_optimize(seq, x0, n_amb, n_sub, n_L, n_H, band_nm,
                                        max_layer_nm, max_total_nm, deadline, 150)
            if thick is not None:
                upd(seq, thick, score)

    # --- 3. Sequence + thickness joint search over random non-alternating stacks ---
    import random
    rng = random.Random(7)
    for trial in range(60):
        if time.time() > deadline - 12.0:
            break
        n = rng.randint(2, min(9, max_layers))
        seq = [rng.choice(['L', 'H']) for _ in range(n)]
        # avoid immediate duplicates of a previously tried exact sequence
        x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq])
        if x0.sum() > max_total_nm:
            x0 = x0 * (max_total_nm / x0.sum())
        x0 = np.clip(x0, 0.0, max_layer_nm)
        labels, thick, score = _nm_optimize_seq(seq, x0, n_amb, n_sub, n_L, n_H,
                                                band_nm, max_layer_nm, max_total_nm,
                                                deadline, 120)
        if thick is not None:
            upd(labels, thick, score)

    # --- 4. Final polish on the global best ---
    if best_labels is not None and time.time() < deadline - 10.0:
        thick, score = _nm_optimize(best_labels, best_thick, n_amb, n_sub, n_L, n_H,
                                    band_nm, max_layer_nm, max_total_nm, deadline, 250)
        if thick is not None:
            upd(best_labels, thick, score)

    # --- Final constraint enforcement ---
    if best_labels is None:
        best_labels = ["L"]
        best_thick = [centre / (4.0 * n_L)]
    best_thick = [max(0.0, min(t, max_layer_nm)) for t in best_thick]
    tot = sum(best_thick)
    if tot > max_total_nm:
        sc = max_total_nm / tot
        best_thick = [t * sc for t in best_thick]
    return best_labels, best_thick

# EVOLVE-BLOCK-END
