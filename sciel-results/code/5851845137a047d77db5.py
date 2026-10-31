"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import Delaunay

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


def cart_to_bary(cart):
    """Convert Cartesian to barycentric coords (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - 0.5 * (y / H)
    u = y / H
    return np.column_stack([u, v])


def clamp_bary(bary, eps=1e-7):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    sums = u + v
    invalid = sums > 1.0
    if np.any(invalid):
        scale = 1.0 / np.maximum(sums[invalid], 1e-12)
        u[invalid] *= scale
        v[invalid] *= scale
    # Push interior points slightly inside
    u = np.clip(u, eps, 1 - 2*eps)
    v = np.clip(v, eps, 1 - 2*eps)
    return np.column_stack([u, v])


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


def _delaunay_smooth_obj(bary_flat, beta=50.0, top_k=10):
    """
    Delaunay-aware smooth objective: focus optimization on the k smallest
    triangles (which are the critical ones), using a soft-min approximation.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Sort areas and focus on the smallest k
    sorted_areas = np.sort(areas)
    critical = sorted_areas[:top_k]
    
    # Soft min over critical triangles
    scaled = -beta * critical
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum / beta


def _neg_soft(bary_flat, beta=50.0, top_k=10):
    return _delaunay_smooth_obj(bary_flat, beta, top_k)


def _neg_hard(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)


def generate_hexagonal_init():
    """
    Generate a structured initial configuration based on hexagonal packing.
    For 11 points: vertices, edge midpoints, and interior points.
    """
    points = []
    # 3 vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.0, 1.0])
    
    # 3 edge midpoints
    points.append([0.5, 0.0])
    points.append([0.5, 0.5])
    points.append([0.0, 0.5])
    
    # Interior points: center and 3 around it
    center = [1/3, 1/3]
    points.append(center)
    
    # 3 more interior points
    points.append([0.25, 0.25])
    points.append([0.25, 0.5])
    points.append([0.5, 0.25])
    
    return np.array(points[:N])


def generate_grid_init():
    """Generate a grid-based initial configuration."""
    points = []
    # 4x4 grid in barycentric, keep only points inside triangle
    for i in range(4):
        for j in range(4):
            u = i / 3.0
            v = j / 3.0
            if u + v <= 1.0:
                points.append([u, v])
    # If we have more than N, take first N
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


def _sa_global(bary, deadline, rng, n_iter=2000, T0=0.05, T1=0.001):
    """
    Simulated annealing global search.
    """
    current = bary.copy()
    current_val = -_neg_hard(current.flatten())
    best = current.copy()
    best_val = current_val
    
    T = T0
    cooling = (T1 / T0) ** (1.0 / max(n_iter, 1))
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        T *= cooling
        
        # Random perturbation
        new_bary = current.copy()
        # Perturb 1-3 random points
        n_perturb = rng.integers(1, 4)
        indices = rng.choice(N, size=n_perturb, replace=False)
        noise = rng.normal(0, 0.05, size=(n_perturb, 2))
        new_bary[indices] += noise
        new_bary = clamp_bary(new_bary)
        
        new_val = -_neg_hard(new_bary.flatten())
        
        # Accept if better or with probability exp((new - cur)/T)
        if new_val > current_val or (T > 1e-6 and rng.random() < np.exp((new_val - current_val) / T)):
            current = new_bary
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best = new_bary.copy()
    
    return best, best_val


def _lbfgs_refine(bary, deadline, max_iter=300, beta=50.0, top_k=10):
    """Refine using L-BFGS-B on the Delaunay-aware smooth objective."""
    try:
        result = minimize(
            _neg_soft,
            bary.flatten(),
            args=(beta, top_k),
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
        val = -_neg_hard(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    rng = np.random.default_rng(42)

    # Phase 1: Generate multiple diverse initial configurations
    inits = [generate_hexagonal_init(), generate_grid_init()]
    for i in range(8):
        inits.append(generate_random_init(rng))
    
    # Evaluate all inits
    for init in inits:
        if time.time() > deadline - 60:
            break
        update_best(init)

    # Phase 2: Simulated annealing from best init
    if best_bary is not None and time.time() < deadline - 40:
        sa_best, sa_val = _sa_global(best_bary, deadline - 40, rng, n_iter=1500)
        update_best(sa_best)

    # Phase 3: Local refinement with L-BFGS on smooth objective
    if best_bary is not None and time.time() < deadline - 20:
        refined = _lbfgs_refine(best_bary, deadline - 20, max_iter=400, beta=50.0, top_k=10)
        update_best(refined)

    # Phase 4: Perturbation + re-refinement cycle
    for i in range(4):
        if time.time() > deadline - 15:
            break
        perturbed = generate_perturbed_init(best_bary, rng, scale=0.03)
        refined = _lbfgs_refine(perturbed, deadline - 15, max_iter=200, beta=80.0, top_k=8)
        update_best(refined)

    # Phase 5: Final polish with Nelder-Mead on hard objective
    if best_bary is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                _neg_hard,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 6: Another SA round from the polished result if time permits
    if best_bary is not None and time.time() < deadline - 10:
        sa_best2, sa_val2 = _sa_global(best_bary, deadline - 10, rng, n_iter=800, T0=0.02, T1=0.001)
        update_best(sa_best2)

    if best_bary is None:
        best_bary = generate_random_init(rng)

    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    return cart
