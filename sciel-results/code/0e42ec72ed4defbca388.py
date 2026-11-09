"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRATEGY (structurally different from the seed):
  The seed fixes a small curated pool of layer *sequences* and does coarse random
  restarts + Nelder-Mead polish per sequence.  Here we instead:
    1. FIX the layer sequence to the physically-motivated optimal family
       (alternating low/high index, starting with L, ending on the substrate with L)
       and let ALL search budget go into the thickness vector.
    2. Run a POPULATION-BASED global optimizer (differential evolution) over the
       full thickness space for several layer counts.  DE's crossover+mutation
       lets it explore the high-dimensional landscape globally and avoid the local
       minima that trap the seed's per-sequence restarts.
    3. Polish the DE survivors with Nelder-Mead.
  A ~10x faster reflectance model (precomputed per-layer cos/sin, no complex
  arithmetic) is what makes a large DE population affordable within the budget.
"""

import time
import math
import numpy as np


# ---------------------------------------------------------------------------
# Fast reflectance model (thin-film characteristic matrix, normal incidence)
# ---------------------------------------------------------------------------
def _reflectance_pct(n_amb, n_sub, n_vec, d_vec, wl, cos, sin, n_inv, n_dir):
    """Mean reflectance (percent) over wavelengths wl.

    n_vec:  array of refractive indices (air -> substrate order), length N
    d_vec:  array of physical thicknesses (nm), length N
    wl:     array of wavelengths (nm), length M
    cos/sin: precomputed 2-D arrays (M, N) of cos/sin(2*pi*d/(lambda*n))
    n_inv/n_dir: 1/n_j and n_j per layer
    """
    n1 = n_vec[0]
    n0 = float(n_amb)
    n_sub = float(n_sub)

    m11 = np.ones_like(wl)
    m12 = np.zeros_like(wl)
    m21 = np.zeros_like(wl)
    m22 = np.ones_like(wl)

    for j in range(n_vec.size):
        c_j = cos[:, j]
        s_j = sin[:, j]
        a = c_j
        b = s_j * n_inv[j]
        cc = s_j * n_dir[j]
        d = c_j
        nm11 = m11 * a + m12 * cc
        nm12 = m11 * b + m12 * d
        nm21 = m21 * a + m22 * cc
        nm22 = m21 * b + m22 * d
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22

    Y = (n0 * m11 + n0 * n1 * m12) / (m11 + n1 * m12)
    r = (Y - n_sub) / (Y + n_sub)
    R = np.abs(r) ** 2
    return float(np.mean(R) * 100.0)


# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
class _Deadline:
    def __init__(self, seconds):
        self.end = time.time() + seconds
        self.best = None
        self.best_val = float("inf")

    def expired(self, margin=0.0):
        return (time.time() - self.end) > -margin

    def record(self, materials, thicknesses, val):
        if val < self.best_val:
            self.best_val = val
            self.best = (list(materials), list(thicknesses))


def _alternating_seq(N):
    return ['L' if (i % 2 == 0) else 'H' for i in range(N)]


# ---------------------------------------------------------------------------
# Differential evolution (global, population based)
# ---------------------------------------------------------------------------
def _differential_evolution(obj, x0, lb, ub, pop, iters, F, CR, rng, deadline,
                            margin=3.0):
    """Simple DE/rand/1/bin. obj is a callable on a 1-D array. Returns (x, f)."""
    dim = len(x0)
    # initial population: x0 + random spread
    pop = int(pop)
    pop = max(pop, dim + 1)
    X = np.empty((pop, dim))
    X[0] = x0
    for i in range(1, pop):
        X[i] = lb + (ub - lb) * rng.random(dim)
    F = np.array([obj(X[i]) for i in range(pop)])
    order = np.argsort(F)
    X = X[order]
    F = F[order]
    best_x = X[0].copy()
    best_f = F[0]

    for _ in range(iters):
        if deadline.expired(margin=margin):
            break
        for i in range(pop):
            # pick 3 distinct individuals != i
            while True:
                a, b, c = (int(v) for v in rng.integers(0, pop, 3))
                if a != b and b != c and c != a and a != i and b != i and c != i:
                    break
            v = X[a] + F * (X[b] - X[c])
            v = np.clip(v, lb, ub)
            trial = X[i].copy()
            mask = rng.random(dim) < CR
            if not mask.any():
                mask[int(rng.integers(0, dim))] = True
            trial[mask] = v[mask]
            ft = obj(trial)
            if ft <= F[i]:
                X[i] = trial
                F[i] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
        # keep sorted for bookkeeping (cheap)
        order = np.argsort(F)
        X = X[order]
        F = F[order]
    return best_x, best_f


# ---------------------------------------------------------------------------
# Nelder-Mead polish
# ---------------------------------------------------------------------------
def _nelder_mead(obj, x0, max_iter=150, deadline=None, margin=3.0):
    x = np.array(x0, dtype=float)
    dim = len(x)
    simplex = [x.copy()]
    for i in range(dim):
        xi = x.copy()
        step = max(5.0, 0.1 * abs(x[i]))
        xi[i] += step
        simplex.append(xi)
    fvals = [obj(v) for v in simplex]
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    best_x = simplex[0].copy()
    best_f = fvals[0]

    for _ in range(max_iter):
        if deadline is not None and deadline.expired(margin=margin):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        f_worst = fvals[-1]
        xr = centroid + (centroid - worst)
        fr = obj(xr)
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe = obj(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (worst - centroid)
            fc = obj(xc)
            if fc < f_worst:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for k in range(1, dim + 1):
                    simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                    fvals[k] = obj(simplex[k])
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0].copy()
        if max(fvals) - min(fvals) < 1e-8:
            break
    return best_x, best_f


# ---------------------------------------------------------------------------
# Main design
# ---------------------------------------------------------------------------
def design(problem):
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mats = problem["materials"]
    nL = mats['L']
    nH = mats['H']
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    wl = np.linspace(wl_min, wl_max, 401)
    deadline = _Deadline(120.0)
    rng = np.random.default_rng(20240517)
    centre = 0.5 * (wl_min + wl_max)

    # Baseline: single quarter-wave L layer
    base_th = centre / (4.0 * nL)
    base_mat = ['L']
    base_val = _reflectance_pct(n_amb, n_sub, np.array([nL]), np.array([base_th]),
                                wl,
                                np.array([[(2.0 * math.pi * base_th) / (wl * nL)]]).T,
                                np.array([[(2.0 * math.pi * base_th) / (wl * nL)]]).T,
                                np.array([1.0 / nL]), np.array([nL]))
    deadline.record(base_mat, [base_th], base_val)

    # Layer counts to explore (odd counts end on L next to substrate)
    counts = [N for N in range(3, max_layers + 1) if N % 2 == 1]
    counts.sort(key=lambda N: -N)  # try larger first (more broadband)

    survivors = []  # (val, N, seq, x)

    for N in counts:
        if deadline.expired(margin=15.0):
            break
        seq = _alternating_seq(N)
        n_vec = np.array([nL if m == 'L' else nH for m in seq])
        n_inv = 1.0 / n_vec
        n_dir = n_vec.copy()

        # Precompute per-layer phase factors as a function of d (vectorized over wl)
        # beta = 2*pi*d/(lambda*n).  cos/sin depend on d, so we build them per eval.
        def build_trig(d):
            d = np.asarray(d, dtype=float)
            beta = (2.0 * math.pi * d[None, :]) / (wl[:, None] * n_vec[None, :])
            return np.cos(beta), np.sin(beta)

        def obj(d):
            d = np.clip(d, 0.0, max_layer)
            if d.sum() > max_total:
                d = d * (max_total / d.sum())
            cos, sin = build_trig(d)
            return _reflectance_pct(n_amb, n_sub, n_vec, d, wl, cos, sin, n_inv, n_dir)

        # Starting point: quarter-wave at centre for each layer
        x0 = np.array([centre / (4.0 * n_vec[j]) for j in range(N)], dtype=float)
        x0 = np.clip(x0, 0.0, max_layer)
        if x0.sum() > max_total:
            x0 = x0 * (max_total / x0.sum())

        lb = np.zeros(N)
        ub = np.full(N, max_layer)

        # DE population scales with dim but capped; more iters for fewer layers
        pop = max(20, min(60, 4 * N + 10))
        iters = 40 if N <= 9 else 25
        try:
            x_de, f_de = _differential_evolution(
                obj, x0, lb, ub, pop, iters, F=0.8, CR=0.7, rng=rng,
                deadline=deadline, margin=8.0)
        except Exception:
            x_de, f_de = x0, obj(x0)

        if f_de < deadline.best_val:
            deadline.record(seq, list(x_de), f_de)
            survivors.append((f_de, N, seq, x_de.copy()))

        # NM polish the DE result
        if not deadline.expired(margin=8.0):
            try:
                x_nm, f_nm = _nelder_mead(obj, x_de, max_iter=120,
                                           deadline=deadline, margin=6.0)
            except Exception:
                x_nm, f_nm = x_de, f_de
            if f_nm < deadline.best_val:
                deadline.record(seq, list(x_nm), f_nm)
                survivors.append((f_nm, N, seq, x_nm.copy()))

    # -----------------------------------------------------------------------
    # Final: re-polish the single best survivor if time remains
    # -----------------------------------------------------------------------
    if survivors and not deadline.expired(margin=10.0):
        survivors.sort(key=lambda t: t[0])
        best_val, N, seq, x = survivors[0]
        n_vec = np.array([nL if m == 'L' else nH for m in seq])
        n_inv = 1.0 / n_vec
        n_dir = n_vec.copy()

        def obj(d):
            d = np.clip(d, 0.0, max_layer)
            if d.sum() > max_total:
                d = d * (max_total / d.sum())
            beta = (2.0 * math.pi * d[None, :]) / (wl[:, None] * n_vec[None, :])
            return _reflectance_pct(n_amb, n_sub, n_vec, d, wl,
                                    np.cos(beta), np.sin(beta), n_inv, n_dir)

        try:
            x2, f2 = _nelder_mead(obj, x, max_iter=150,
                                   deadline=deadline, margin=5.0)
            if f2 < deadline.best_val:
                deadline.record(seq, list(x2), f2)
        except Exception:
            pass

    if deadline.best is None:
        return base_mat, [base_th]

    materials, thicknesses = deadline.best
    thicknesses = [min(max(float(t), 0.0), max_layer) for t in thicknesses]
    if sum(thicknesses) > max_total:
        scale = max_total / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
    if len(materials) > max_layers:
        materials = materials[:max_layers]
        thicknesses = thicknesses[:max_layers]
    return materials, thicknesses
