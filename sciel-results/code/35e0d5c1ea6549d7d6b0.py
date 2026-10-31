import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

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

def clamp_bary(bary, eps=1e-9):
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

def get_min_area(bary):
    """Get the minimum area for a barycentric configuration."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def epigraph_obj(x_flat):
    """
    Epigraph formulation: maximize t subject to all areas >= t.
    x_flat = [bary (2N), t]
    Minimize: -t + penalty for violations
    """
    bary = x_flat[:2*N].reshape(N, 2)
    t = x_flat[2*N]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas below t
    violations = np.maximum(0, t - areas)
    penalty = 100.0 * np.sum(violations**2)
    
    # Objective: minimize -t + penalty
    return -t + penalty

def epigraph_grad(x_flat):
    """Gradient for epigraph objective."""
    bary = x_flat[:2*N].reshape(N, 2)
    t = x_flat[2*N]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Gradient of penalty w.r.t. bary
    grad_bary = np.zeros_like(bary)
    for i in range(len(areas)):
        if areas[i] < t:
            violation = t - areas[i]
            # Gradient of (t - areas[i])^2 w.r.t. bary
            # d/dx (t - A(x))^2 = -2(t - A(x)) * dA/dx
            # We need dA/dx for the i-th triangle
            i0, i1, i2 = tri_idx[i]
            # Area = 0.5 * |det|, but we use abs(det) so gradient is sign-dependent
            # For simplicity, use numerical gradient
            pass
    
    # Gradient w.r.t. t: -1 + 2*100*sum(max(0, t-areas))
    violations = np.maximum(0, t - areas)
    grad_t = -1.0 + 2.0 * 100.0 * np.sum(violations)
    
    # For now, use finite differences for bary gradient
    eps = 1e-7
    grad_flat = np.zeros_like(x_flat)
    for j in range(2*N):
        x_plus = x_flat.copy()
        x_plus[j] += eps
        f_plus = epigraph_obj(x_plus)
        grad_flat[j] = (f_plus - (-t + 100.0 * np.sum(np.maximum(0, t - areas)**2))) / eps
    
    grad_flat[2*N] = grad_t
    
    return grad_flat

def refine_epigraph(bary_init, time_limit=10.0):
    """
    Refine using epigraph formulation.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    t_init = get_min_area(bary_current)
    
    # Start with t slightly below current min area
    t_current = t_init * 0.9
    
    x_current = np.concatenate([bary_current.flatten(), [t_current]])
    
    # Multi-stage refinement with increasing t targets
    for stage in range(3):
        if time.time() - start_time > time_limit:
            break
        
        def obj(x_flat):
            return epigraph_obj(x_flat)
        
        try:
            result = minimize(
                obj,
                x_current,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            x_new = result.x
            bary_new = clamp_bary(x_new[:2*N].reshape(N, 2))
            t_new = x_new[2*N]
            
            # Check if this is valid (all areas >= t_new - small tol)
            areas_new = calc_areas_bary(bary_new)
            if np.min(areas_new) >= t_new - 1e-8:
                x_current = x_new
                bary_current = bary_new
                t_current = t_new
        except Exception:
            pass
    
    return bary_current

def generate_hexagonal_init(k=3, seed=0):
    """
    Generate a hexagonal-like lattice of points.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(k + 1):
        for j in range(k - i + 1):
            u = i / k + rng.normal(0, 0.01)
            v = j / k + rng.normal(0, 0.01)
            pts.append([u, v])
    
    pts = np.array(pts)
    pts = clamp_bary(pts)
    
    if len(pts) < N:
        while len(pts) < N:
            u = rng.random()
            v = rng.random() * (1 - u)
            pt = np.array([u, v])
            if len(pts) > 0:
                dists = np.linalg.norm(pts - pt, axis=1)
                if np.min(dists) > 0.1:
                    pts = np.vstack([pts, [pt]])
            else:
                pts = np.vstack([pts, [pt]])
    elif len(pts) > N:
        pts = pts[:N]
        
    return pts

def basin_hopping_step(bary, rng, scale=0.05):
    """
    Generate a perturbed version of bary for basin hopping.
    """
    bary_perturbed = bary.copy()
    # Add random Gaussian noise
    noise = rng.normal(0, scale, size=bary.shape)
    bary_perturbed = bary_perturbed + noise
    bary_perturbed = clamp_bary(bary_perturbed)
    return bary_perturbed

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate initial candidates
    candidates = []
    candidates.append(generate_hexagonal_init(k=3, seed=1))
    candidates.append(generate_hexagonal_init(k=4, seed=2))
    candidates.append(generate_hexagonal_init(k=3, seed=3))
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random() * (1 - u)
                pt = np.array([u, v])
                if i == 0:
                    break
                dists = np.linalg.norm(pts[:i] - pt, axis=1)
                if np.min(dists) > 0.15:
                    break
            pts[i] = pt
        candidates.append(pts)

    # 2. Refine initial candidates with epigraph
    # Allocate 30% of time to initial refinement
    t_init = (TIME_LIMIT * 0.3) / len(candidates)
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        refined = refine_epigraph(cand, time_limit=t_init)
        update_best(refined)

    # 3. Basin hopping for global search
    # Allocate 60% of time to basin hopping
    t_bh = TIME_LIMIT * 0.6
    bh_start = time.time()
    
    current_bary = best_bary.copy()
    current_val = best_val
    
    # Basin hopping parameters
    max_iterations = 50
    scales = [0.1, 0.05, 0.02, 0.01]
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 20:
            break
        
        # Choose perturbation scale (decrease over time)
        scale_idx = min(iteration // 10, len(scales) - 1)
        scale = scales[scale_idx]
        
        # Generate perturbed candidate
        bary_perturbed = basin_hopping_step(current_bary, rng, scale=scale)
        
        # Refine the perturbed candidate
        t_per_step = min(t_bh / max_iterations, 5.0)
        refined = refine_epigraph(bary_perturbed, time_limit=t_per_step)
        val = get_min_area(refined)
        
        # Accept if improved or with some probability (Metropolis-like)
        if val > current_val:
            current_bary = refined
            current_val = val
            update_best(refined)
        elif rng.random() < np.exp(-(current_val - val) / 0.001):
            # Accept worse solution with some probability
            current_bary = refined
            current_val = val
    
    # 4. Final Polish
    # Allocate remaining time to polish the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        try:
            refined_best = refine_epigraph(best_bary, time_limit=remaining)
            update_best(refined_best)
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
