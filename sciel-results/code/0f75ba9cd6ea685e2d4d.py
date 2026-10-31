"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.

Structural approach: Delaunay-aware smooth surrogate + multi-start L-BFGS + coordinate-descent polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)


def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])


def cart_to_bary(cart):
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])


def clamp_bary(bary, eps=1e-7):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])


def calc_areas_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return np.abs(det)


def hard_min_obj(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)


def soft_min_obj(bary_flat, beta=100.0):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    m = np.max(scaled)
    if m > 700:
        return -np.min(areas)
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    soft_min = -log_sum / beta
    return -soft_min


def delaunay_smooth_obj(bary_flat, beta=60.0):
    """
    Delaunay-aware smooth surrogate.
    - Compute Delaunay triangulation of the point set.
    - For each Delaunay triangle, compute its area.
    - Objective = -soft_min over Delaunay areas + small penalty for non-Delaunay small triangles.
    This focuses the optimizer on the triangles that actually bound the minimum.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    areas_all = calc_areas_bary(bary)

    try:
        tri = Delaunay(cart)
        d_idx = tri.simplices  # (M, 3)
        # Ensure consistent orientation (area positive)
        d_areas = np.array([areas_all[_find_tri_index(i, j, k)]
                            for (i, j, k) in d_idx])
    except Exception:
        # Fall back to all triangles
        d_areas = areas_all

    # Soft min over the Delaunay (small) triangles
    scaled = -beta * d_areas
    m = np.max(scaled)
    if m > 700:
        return -np.min(d_areas)
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    soft_min = -log_sum / beta

    # Small penalty: any triangle (even non-Delaunay) that is very small
    # This prevents the optimizer from ignoring a tiny non-Delaunay triangle.
    all_min = np.min(areas_all)
    penalty = max(0.0, (0.5 * np.median(areas_all)) - all_min) ** 2 * 0.01

    return -soft_min + penalty


def _find_tri_index(i, j, k):
    # Find index in tri_idx for the combination {i,j,k}
    target = frozenset((i, j, k))
    for idx, t in enumerate(tri_idx):
        if frozenset(tuple(t)) == target:
            return idx
    # fallback: linear search
    for idx in range(len(tri_idx)):
        if frozenset(tuple(tri_idx[idx])) == target:
            return idx
    return 0


# Precompute a fast lookup: frozenset -> index
_TRI_LOOKUP = {frozenset(tuple(t)): i for i, t in enumerate(tri_idx)}


def _find_tri_index_fast(i, j, k):
    return _TRI_LOOKUP[frozenset((i, j, k))]


def delaunay_smooth_obj_fast(bary_flat, beta=60.0):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    areas_all = calc_areas_bary(bary)
    try:
        tri = Delaunay(cart)
        d_areas = np.array([areas_all[_TRI_LOOKUP[frozenset((i, j, k))]]
                            for (i, j, k) in tri.simplices])
    except Exception:
        d_areas = areas_all
    scaled = -beta * d_areas
    m = np.max(scaled)
    if m > 700:
        return -np.min(d_areas)
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    soft_min = -log_sum / beta
    all_min = np.min(areas_all)
    med = np.median(areas_all)
    penalty = max(0.0, 0.5 * med - all_min) ** 2 * 0.01
    return -soft_min + penalty


def generate_hexagonal_init():
    best_bary = None
    best_val = -np.inf
    for s in np.linspace(0.08, 0.5, 25):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s * 0.866
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary


def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])


def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)


def _local_polish_bary(bary, max_iter=400, deadline=None):
    """
    Coordinate-descent polish: for each point, try small steps in u and v.
    Accepts a step only if it improves the hard-min objective.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary
    current_bary = clamp_bary(bary.copy())
    current_val = -hard_min_obj(current_bary.flatten())
    rng = np.random.default_rng(123)
    steps = [0.002, -0.002, 0.005, -0.005, 0.01, -0.01, 0.0005, -0.0005]
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        improved = False
        order = list(range(N))
        rng.shuffle(order)
        for i in order:
            for du in steps:
                trial = current_bary.copy()
                trial[i, 0] += du
                trial = clamp_bary(trial)
                val = -hard_min_obj(trial.flatten())
                if val > current_val:
                    current_bary = trial
                    current_val = val
                    improved = True
            for dv in steps:
                trial = current_bary.copy()
                trial[i, 1] += dv
                trial = clamp_bary(trial)
                val = -hard_min_obj(trial.flatten())
                if val > current_val:
                    current_bary = trial
                    current_val = val
                    improved = True
        if not improved:
            break
    return current_bary


def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize(bary_init, use_delaunay=True, beta1=60.0, beta2=200.0,
                     maxiter1=250, maxiter2=350):
        if time.time() > deadline - 12:
            return
        try:
            obj1 = (lambda x: delaunay_smooth_obj_fast(x, beta=beta1)) if use_delaunay \
                else (lambda x: soft_min_obj(x, beta=beta1))
            bounds = [(0, 1)] * (N * 2)
            r1 = minimize(obj1, bary_init.flatten(), method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8})
            if time.time() > deadline - 6:
                update_best(r1.x.reshape(N, 2))
                return
            obj2 = (lambda x: delaunay_smooth_obj_fast(x, beta=beta2)) if use_delaunay \
                else (lambda x: soft_min_obj(x, beta=beta2))
            r2 = minimize(obj2, r1.x, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10})
            update_best(r2.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 1: multi-start with Delaunay-aware objective
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, use_delaunay=True, beta1=60.0, beta2=200.0,
                 maxiter1=300, maxiter2=400)

    for i in range(12):
        if time.time() > deadline - 50:
            break
        rand_init = generate_random_init(seed=100 + i * 7)
        try_optimize(rand_init, use_delaunay=True, beta1=60.0, beta2=180.0,
                     maxiter1=150, maxiter2=200)

    # Also a couple of plain soft-min starts for diversity
    for i in range(3):
        if time.time() > deadline - 40:
            break
        rand_init = generate_random_init(seed=300 + i * 11)
        try_optimize(rand_init, use_delaunay=False, beta1=50.0, beta2=200.0,
                     maxiter1=150, maxiter2=200)

    # Phase 2: perturb best + re-optimize (local refinement)
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 25:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=800 + i)
            try_optimize(perturbed, use_delaunay=True, beta1=100.0, beta2=300.0,
                         maxiter1=200, maxiter2=300)

    # Phase 3: Nelder-Mead polish on hard min
    if best_bary is not None and time.time() < deadline - 12:
        try:
            r = minimize(hard_min_obj, best_bary.flatten(), method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True})
            update_best(r.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 4: tight coordinate-descent polish
    if best_bary is not None and time.time() < deadline - 6:
        polished = _local_polish_bary(best_bary, max_iter=500, deadline=deadline)
        update_best(polished)

    # Phase 5: final NM + coordinate polish
    if best_bary is not None and time.time() < deadline - 4:
        try:
            r = minimize(hard_min_obj, best_bary.flatten(), method='Nelder-Mead',
                         options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14,
                                  'adaptive': True})
            update_best(r.x.reshape(N, 2))
        except Exception:
            pass
        if time.time() < deadline - 2:
            polished = _local_polish_bary(best_bary, max_iter=300, deadline=deadline)
            update_best(polished)

    if best_bary is None:
        best_bary = generate_random_init(seed=1)
    best_bary = clamp_bary(best_bary)
    return bary_to_cart(best_bary)


if __name__ == "__main__":
    pts = construct()
    print("Points:")
    print(pts)
    w = pts[:, 1] / H
    v = pts[:, 0] - 0.5 * w
    u = 1 - v - w
    bary = np.column_stack([u, v])
    areas = calc_areas_bary(bary)
    print("Min Area (normalized):", np.min(areas))
    print("Max Area (normalized):", np.max(areas))
