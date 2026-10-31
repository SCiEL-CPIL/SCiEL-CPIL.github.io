"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian for final output
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
# We need all combinations of 3 indices from N
# C(11, 3) = 165
tri_indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            tri_indices.append((i, j, k))
tri_idx = np.array(tri_indices)  # shape (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    bary: (N, 2) array where column 0 is u, column 1 is v. w = 1 - u - v.
    
    The area of a triangle formed by points with barycentric coords P_i, P_j, P_k 
    is proportional to the determinant of their barycentric matrix.
    Specifically, Area(P_i, P_j, P_k) / Area(Reference) = |det(M)|
    where M = [[u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]]
    
    Since we want to maximize the smallest area, and the reference area is constant,
    maximizing the min determinant is equivalent.
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
    
    # Determinant calculation:
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
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
        # Scale down u and v for invalid points
        # We want u' + v' = 1 - eps, preserving ratio
        target_sum = 1 - eps
        scale = target_sum / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling (safety)
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    Soft min is a smooth approximation of the min function.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max (which is the min area term in the exponent if we flip sign correctly)
    # Let f(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to minimize -f(x) = 1/beta * log(sum(exp(-beta * x)))
    # Let's compute -soft_min directly to minimize.
    
    # scaled = -beta * areas
    # max_scaled = np.max(scaled) -> this corresponds to the MIN area
    # log_sum = max_scaled + log(sum(exp(scaled - max_scaled)))
    # soft_min = -log_sum / beta
    
    # We want to MINIMIZE the objective.
    # Objective = - soft_min = log_sum / beta
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    # Handle potential overflow/underflow in exp
    # scaled - max_scaled is <= 0, so exp is <= 1. Safe.
    exp_terms = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp < 1e-300:
        return -np.min(areas) # Fallback
        
    log_sum = max_scaled + np.log(sum_exp)
    
    # Objective to minimize is -soft_min
    return log_sum / beta

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    Returns negative of the minimum area (to be minimized).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*V0 + v*V1 + w*V2
    V0=(0,0), V1=(1,0), V2=(0.5, H)
    x = u*0 + v*1 + w*0.5 = v + 0.5*w
    y = u*0 + v*0 + w*H = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration in barycentric coords.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try different grid spacings
    for s in np.linspace(0.1, 0.4, 10):
        points = []
        for i in range(int(1/s) + 2):
            for j in range(int(1/s) + 2):
                u = i * s
                v = j * s
                if u >= 0 and v >= 0 and u + v <= 1:
                    if len(points) < N:
                        points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            bary = clamp_bary(bary)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        # Fallback to random
        best_bary = generate_random_init(seed=42)
        
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    # Uniform sampling in triangle:
    # u = r1 * sqrt(r2), v = r1 * (1 - sqrt(r2)) ? No, that's for area uniform.
    # Simple method: u = r1, v = r2 * (1 - u)
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
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_bary = None
    best_score = -np.inf  # We maximize min area, so track min area directly for comparison
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        # Calculate the actual min area (hard min)
        min_area = np.min(calc_areas_bary(bary))
        
        if min_area > best_score:
            best_score = min_area
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
        if time.time() > deadline - 30:
            break
        rand_init = generate_random_init(seed=100 + i * 7)
        try_optimize_l_bfgs(rand_init, beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200)
        
    # 3. Perturb Best
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 20:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=500 + i)
            try_optimize_l_bfgs(perturbed, beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
    
    # --- Phase 2: Nelder-Mead Polish ---
    if best_bary is not None and time.time() < deadline - 10:
        try_optimize_nm(best_bary, maxiter=3000)
        
        # Re-perturb and NM if time allows
        for i in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.01, seed=900 + i)
            try_optimize_nm(perturbed, maxiter=2000)

    # --- Phase 3: Final Coordinate Descent / Small Step NM ---
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
