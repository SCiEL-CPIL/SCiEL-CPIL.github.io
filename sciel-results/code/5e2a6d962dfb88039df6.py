"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
NUM_TRIS = len(tri_indices)


def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])


def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]

    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]

    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]

    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)

    return np.abs(det)


def clamp_bary(bary, eps=1e-6):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    return np.column_stack([u, v])


def hard_min_obj(bary_flat):
    """Objective: negative of minimum triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)


def soft_min_obj(bary_flat, beta=100.0):
    """Smooth approximation of the min using log-sum-exp."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum / beta


def generate_hexagonal_init():
    """
    Generate a structured initial configuration based on hexagonal packing.
    For 11 points, use a 3x3 grid plus 2 extra points.
    """
    # 3x3 grid in barycentric coordinates
    # Row 0 (bottom): u=0, v varies
    # Row 1 (middle): u=0.33, v varies
    # Row 2 (top): u=0.66, v varies
    
    points = []
    # Bottom row: 3 points
    for i in range(3):
        v = (i + 0.5) / 3.0
        points.append([0.0, v])
    
    # Middle row: 3 points
    for i in range(3):
        u = 0.33
        v = (i + 0.5) / 3.0 * (1 - u)
        points.append([u, v])
    
    # Top row: 3 points
    for i in range(3):
        u = 0.66
        v = (i + 0.5) / 3.0 * (1 - u)
        points.append([u, v])
    
    # Two extra points near center
    points.append([0.33, 0.22])
    points.append([0.55, 0.15])
    
    return np.array(points[:N])


def generate_random_init(rng):
    """Generate random initial configuration."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])


def generate_perturbed_init(base, rng, scale=0.1):
    """Generate perturbed version of a base configuration."""
    bary = base.copy()
    noise = rng.normal(0, scale, size=bary.shape)
    bary = bary + noise
    return clamp_bary(bary)


def refine_local(bary, deadline, max_iter=200):
    """Refine a configuration using L-BFGS-B on soft min."""
    try:
        result = minimize(
            soft_min_obj,
            bary.flatten(),
            args=(150.0,),
            method='L-BFGS-B',
            bounds=[(0, 1)] * (N * 2),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() < deadline:
            return clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass
    return bary


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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

    # Phase 1: Try multiple structured and random initializations
    rng = np.random.default_rng(42)
    
    # Start with hexagonal init
    init_bary = generate_hexagonal_init()
    update_best(init_bary)
    
    # Try several random inits
    for i in range(10):
        if time.time() > deadline - 30:
            break
        random_bary = generate_random_init(rng)
        update_best(random_bary)
    
    # Phase 2: Local refinement of best candidates
    if best_bary is not None:
        refined = refine_local(best_bary, deadline, max_iter=300)
        update_best(refined)
    
    # Phase 3: Perturbation and re-refinement
    for i in range(5):
        if time.time() > deadline - 20:
            break
        perturbed = generate_perturbed_init(best_bary, rng, scale=0.05)
        refined = refine_local(perturbed, deadline, max_iter=200)
        update_best(refined)
    
    # Phase 4: Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    if best_bary is None:
        rng = np.random.default_rng(0)
        best_bary = generate_random_init(rng)

    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    return cart
