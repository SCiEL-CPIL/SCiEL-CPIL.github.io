"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_structured_init():
    """
    Generate a structured initial configuration.
    Uses a mix of lattice points and boundary points to ensure good spread.
    """
    # 1. Centroid
    # 2. Midpoints of edges
    # 3. Corner points (slightly inside)
    # 4. Grid points
    
    eps = 0.01
    points = []
    
    # Corners (pushed in)
    points.append([eps, eps])
    points.append([1 - eps, eps])
    points.append([eps, 1 - eps])
    
    # Midpoints of edges
    points.append([0.5, eps])
    points.append([0.5, 1 - 0.5 - eps]) # Midpoint of left edge? No, barycentric (u,v)
    # Edge 1 (bottom): u=0, v in [0,1]. Midpoint v=0.5 -> (0, 0.5)
    points.append([0, 0.5])
    # Edge 2 (right): v=0, u in [0,1]. Midpoint u=0.5 -> (0.5, 0)
    points.append([0.5, 0])
    # Edge 3 (top): u+v=1. Midpoint u=0.5, v=0.5 -> (0.5, 0.5)
    points.append([0.5, 0.5])
    
    # Centroid
    points.append([1/3, 1/3])
    
    # Additional grid points to fill space
    # We need 11 points. We have 7. Need 4 more.
    # Try a few interior points
    candidates = [
        [0.2, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.4, 0.4],
        [0.6, 0.3],
        [0.3, 0.6]
    ]
    
    for p in candidates:
        if len(points) >= N:
            break
        # Check if close to existing points
        is_new = True
        for existing in points:
            if np.linalg.norm(np.array(p) - np.array(existing)) < 0.1:
                is_new = False
                break
        if is_new:
            points.append(p)
            
    # If still not enough, just take random valid ones
    rng = np.random.default_rng(123)
    while len(points) < N:
        u = rng.random()
        v = rng.random() * (1 - u)
        points.append([u, v])
        
    return np.array(points[:N])

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize_multi_stage(bary_init, stages=None):
        """
        Multi-stage optimization with increasing beta (annealing).
        """
        if time.time() > deadline - 10:
            return
        
        if stages is None:
            # (beta, maxiter)
            stages = [
                (20.0, 100),
                (100.0, 200),
                (500.0, 300),
                (2000.0, 500)
            ]
        
        current_x = bary_init.flatten()
        
        for beta, maxiter in stages:
            if time.time() > deadline - 5:
                break
            
            try:
                def obj(x):
                    return soft_min_obj(x, beta=beta)
                
                bounds = [(0, 1)] * (N * 2)
                
                result = minimize(
                    obj,
                    current_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                current_x = result.x
                update_best(result.x.reshape(N, 2))
                
            except Exception:
                pass

    # 1. Start with structured initialization
    struct_init = generate_structured_init()
    try_optimize_multi_stage(struct_init)
    
    # 2. Try multiple random initializations
    # Increase number of restarts to improve global search
    num_random_starts = 15
    for i in range(num_random_starts):
        if time.time() > deadline - 30:
            break
        rng_seed = 1000 + i
        rand_init = generate_random_init(seed=rng_seed)
        # Use fewer iterations for random starts to allow more of them
        stages = [
            (50.0, 100),
            (200.0, 200),
            (1000.0, 300)
        ]
        try_optimize_multi_stage(rand_init, stages=stages)
        
    # 3. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(10):
            if time.time() > deadline - 20:
                break
            # Use smaller noise for fine-tuning
            noise = 0.01 + 0.01 * i
            perturbed = generate_perturbed_init(best_bary, noise_scale=noise, seed=2000 + i)
            stages = [
                (100.0, 100),
                (500.0, 200),
                (2000.0, 300)
            ]
            try_optimize_multi_stage(perturbed, stages=stages)
    
    # 4. Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 10:
        try:
            bary_nm = best_bary.copy()
            # Ensure strictly inside
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # 5. Additional local search with smaller steps
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # Try multiple small perturbations and keep improvements
            for i in range(30):
                if time.time() > deadline - 3:
                    break
                perturbed = generate_perturbed_init(best_bary, noise_scale=0.005, seed=3000 + i)
                if update_best(perturbed):
                    pass  # Keep the improvement
        except Exception:
            pass
    
    # If all failed, generate a random valid set
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    # Convert to Cartesian
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Final safety check: ensure all points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
            
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
             cart[i, 1] = 1e-9
             
    return cart
