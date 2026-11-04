"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286
TRI_ARR = np.array(TRI_INDICES, dtype=np.int32)  # (286, 3)

def _all_tri_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    i, j, k = TRI_ARR[:,0], TRI_ARR[:,1], TRI_ARR[:,2]
    a = pts[i]
    b = pts[j]
    c = pts[k]
    # 0.5 * |cross(b-a, c-a)|
    v1 = b - a
    v2 = c - a
    areas = 0.5 * np.abs(v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0])
    return areas

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min_ratio(pts, temperature):
    """Smooth surrogate: soft-min of triangle areas / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -1e9
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return -1e9
    except Exception:
        return -1e9
    areas = _all_tri_areas(pts)
    areas = np.maximum(areas, 1e-15)
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    return soft_min_area / hull_area

def _neg_soft_min(x, temperature):
    pts = x.reshape(N, 2)
    r = _soft_min_ratio(pts, temperature)
    if r < -1e8:
        return 1e8
    return -r

def _neg_score(x):
    pts = x.reshape(N, 2)
    r = _score(pts)
    return -r

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    pts = [[0.0, 0.0]]
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        n_on_ring = min(remaining, max(3, (n-1)//n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    if remaining > 0:
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        pts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    result = np.vstack(pts)
    return result[:n]

def _halton(n, base):
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        idx = i + 1
        while idx > 0:
            f /= base
            idx = (idx - 1) // base
        result[i] = f
    return result

def _halton_pts(n, bases=(2,3)):
    return np.column_stack([_halton(n, b) for b in bases])

def _structured_grid(n):
    pts = [[0.5, 0.5]]
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.5 + 0.35*np.cos(angle), 0.5 + 0.35*np.sin(angle)])
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        pts.append([0.5 + 0.45*np.cos(angle), 0.5 + 0.45*np.sin(angle)])
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.5 + 0.48*np.cos(angle + np.pi/8), 0.5 + 0.48*np.sin(angle + np.pi/8)])
    return np.array(pts)

def _polish(x0, time_budget, rng):
    """Multi-phase polish: annealed soft-min L-BFGS-B, then Nelder-Mead, then local search."""
    start_time = time.time()
    best_x = x0.copy()
    best_score = _score(x0.reshape(N, 2))

    # Phase 1: annealed soft-min with L-BFGS-B
    for temp in [0.05, 0.02, 0.01, 0.005, 0.002]:
        if time.time() - start_time > time_budget - 3:
            break
        try:
            res = minimize(lambda x: _neg_soft_min(x, temp), best_x, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            cand = res.x
            cand_score = _score(cand.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_x = cand.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead on exact objective
    if time.time() - start_time < time_budget - 2:
        try:
            res = minimize(_neg_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10})
            cand = res.x
            cand_score = _score(cand.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_x = cand.copy()
        except Exception:
            pass

    # Phase 3: local perturbation on min triangle
    if time.time() - start_time < time_budget - 1:
        best_pts = best_x.reshape(N, 2)
        for _ in range(60):
            if time.time() - start_time > time_budget - 0.5:
                break
            areas = _all_tri_areas(best_pts)
            min_idx = np.argmin(areas)
            i, j, k = TRI_INDICES[min_idx]
            for pidx in [i, j, k]:
                for scale in [0.01, 0.005, 0.002]:
                    if time.time() - start_time > time_budget - 0.3:
                        break
                    trial = best_pts.copy()
                    trial[pidx] += rng.normal(0, scale, size=2)
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_x = trial.flatten().copy()
                        best_pts = trial
                        break
    return best_x.reshape(N, 2), best_score

def construct():
    start_time = time.time()
    total_budget = 145.0  # safety margin for 150s limit
    rng = np.random.RandomState(42)

    def _time_left():
        return total_budget - (time.time() - start_time)

    # Build diverse initial population
    pop = []
    pop.append(_regular_polygon(N, 1.0))
    pop.append(_concentric_rings(N, 3))
    pop.append(_concentric_rings(N, 4))
    pop.append(_halton_pts(N, (2,3)))
    pop.append(_halton_pts(N, (3,5)))
    pop.append(_structured_grid(N))
    for _ in range(6):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pop.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]))

    # Evaluate population
    pop_scores = []
    for p in pop:
        if _time_left() < 5:
            break
        try:
            pop_scores.append(_score(p))
        except Exception:
            pop_scores.append(0.0)
    best_idx = int(np.argmax(pop_scores))
    best_score = pop_scores[best_idx]
    best_pts = pop[best_idx].copy()

    # DE global search: bounds [-1.5, 1.5] per coordinate
    bounds = [(-1.5, 1.5)] * (2*N)

    # Use DE with the soft-min surrogate for speed, then polish
    if _time_left() > 30:
        de_budget = min(60.0, _time_left() - 20.0)
        def de_obj(x):
            if time.time() - start_time > total_budget - 15:
                return 1e8
            pts = x.reshape(N, 2)
            r = _soft_min_ratio(pts, 0.02)
            if r < -1e8:
                return 1e8
            return -r
        try:
            de_res = differential_evolution(
                de_obj, bounds,
                popsize=8, maxiter=40, tol=1e-6,
                seed=42, polish=False, init='sobol',
                mutation=(0.4, 1.0), recombination=0.7,
                updating='deferred',
            )
            if de_res.fun < 1e7:
                cand = de_res.x
                cand_score = _score(cand.reshape(N, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand.reshape(N, 2).copy()
        except Exception:
            pass

    # Polish the best found (and a couple of other top candidates if time)
    if best_pts is not None and _time_left() > 5:
        final_time = _time_left() - 1.0
        if final_time > 5.0:
            try:
                fp, fs = _polish(best_pts.flatten(), final_time, rng)
                if fs > best_score:
                    best_score = fs
                    best_pts = fp.copy()
            except Exception:
                pass

    if best_pts is None:
        best_pts = _regular_polygon(N, 1.0)

    return best_pts
# EVOLVE-BLOCK-END
