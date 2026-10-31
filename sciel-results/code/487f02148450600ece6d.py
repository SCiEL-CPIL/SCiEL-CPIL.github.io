"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
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

def anneal_obj(bary_flat, T):
    """
    Simulated annealing objective: minimize -min_area + T * (some penalty or just the min area).
    Actually, for SA we want to accept worse moves. So we just return the value to minimize.
    Value = -min_area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def simulated_annealing(bary_init, T0=0.1, T_end=0.001, n_iter=2000, deadline=None):
    """
    Simulated Annealing to escape local minima.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary_init
        
    current_bary = clamp_bary(bary_init)
    current_val = anneal_obj(current_bary.flatten(), T0)
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
            
        T = T0 * (T_end / T0) ** (i / n_iter)
        
        # Generate a random perturbation
        # Scale noise by T to allow larger jumps early on
        noise_scale = 0.05 * np.sqrt(T / T0) + 0.01
        perturbed_bary = current_bary + rng.normal(0, noise_scale, current_bary.shape)
        perturbed_bary = clamp_bary(perturbed_bary)
        
        new_val = anneal_obj(perturbed_bary.flatten(), T)
        
        delta = new_val - current_val
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_bary = perturbed_bary
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_bary = current_bary.copy()
                
    return best_bary

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. We use 140s deadline to be safe.
    deadline = start_time + 140.0
    
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

    def try_optimize_l_bfgs(bary_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage L-BFGS optimization.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration with smoother landscape
            def obj1(x):
                return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation with sharper landscape
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    def try_optimize_nm(bary_init, maxiter=2000):
        """
        Nelder-Mead optimization on hard min.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            result = minimize(
                hard_min_obj,
                bary_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # --- Phase 1: Multi-start L-BFGS ---
    
    # 1. Structured Init
    hex_init = generate_hexagonal_init()
    try_optimize_l_bfgs(hex_init, beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400)
    
    # 2. Random Inits
    num_random_starts = 10
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rand_init = generate_random_init(seed=100 + i * 7)
        try_optimize_l_bfgs(rand_init, beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200)
        
    # --- Phase 2: Simulated Annealing to escape local minima ---
    # This is the structural change: using SA to jump between basins before final polish.
    if best_bary is not None:
        # Run SA starting from the best found so far
        sa_result = simulated_annealing(best_bary, T0=0.2, T_end=0.01, n_iter=3000, deadline=deadline)
        update_best(sa_result)
        
        # Try SA from a few random starts as well if time permits
        for i in range(3):
            if time.time() > deadline - 20:
                break
            rand_start = generate_random_init(seed=500 + i * 13)
            # Quick optimize to get to a basin first
            try_optimize_l_bfgs(rand_start, beta1=50.0, beta2=150.0, maxiter1=100, maxiter2=100)
            if time.time() > deadline - 15:
                break
            # Then SA
            sa_result = simulated_annealing(best_bary, T0=0.1, T_end=0.005, n_iter=2000, deadline=deadline)
            update_best(sa_result)

    # --- Phase 3: Perturb Best and Re-optimize (Local Refinement) ---
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 15:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=800 + i)
            try_optimize_l_bfgs(perturbed, beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
    
    # --- Phase 4: Nelder-Mead Polish ---
    if best_bary is not None and time.time() < deadline - 10:
        try_optimize_nm(best_bary, maxiter=3000)
        
        # Re-perturb and NM if time allows
        for i in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.01, seed=900 + i)
            try_optimize_nm(perturbed, maxiter=2000)

    # --- Phase 5: Final Coordinate Descent / Small Step NM ---
    if best_bary is not None and time.time() < deadline - 5:
        # Use a very tight NM polish
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # Ensure we have a valid result
    if best_bary is None:
        best_bary = generate_random_init(seed=1)
        
    # Final clamp
    best_bary = clamp_bary(best_bary)
    
    # Convert to Cartesian
    cart_pts = bary_to_cart(best_bary)
    
    return cart_pts

# Test run if executed directly
if __name__ == "__main__":
    pts = construct()
    print("Points:")
    print(pts)
    # Check validity
    for p in pts:
        if p[0] < -1e-9 or p[1] < -1e-9 or p[0] > 1 + 1e-9:
            print("Invalid point:", p)
        if p[1] > np.sqrt(3) * p[0] + 1e-9 or p[1] > np.sqrt(3) * (1 - p[0]) + 1e-9:
            print("Invalid point (triangle bound):", p)
    # Calculate score
    bary = np.zeros((N, 2))
    # Inverse of bary_to_cart for debugging score
    # x = v + 0.5*w, y = w*H => w = y/H, v = x - 0.5*w, u = 1 - v - w
    w = pts[:, 1] / H
    v = pts[:, 0] - 0.5 * w
    u = 1 - v - w
    bary = np.column_stack([u, v])
    areas = calc_areas_bary(bary)
    print("Min Area (normalized):", np.min(areas))
    print("Max Area (normalized):", np.max(areas))
