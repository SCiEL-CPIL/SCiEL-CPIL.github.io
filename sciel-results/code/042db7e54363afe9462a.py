import numpy as np
from itertools import combinations
import time
from scipy.optimize import differential_evolution

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

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
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

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = min_area(bary)
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

def objective(x):
    """
    Objective function for Differential Evolution.
    We want to maximize min_area, so we minimize -min_area.
    We add a penalty for points outside the triangle to handle boundaries softly 
    or rely on the bounds if we use Cartesian, but here we use Barycentric.
    In Barycentric coords, valid region is u>=0, v>=0, u+v<=1.
    DE works on a hyper-rectangle. We map the hyper-rectangle to the triangle?
    No, DE expects bounds. We can set bounds for u in [0,1] and v in [0,1].
    Then we must penalize u+v > 1.
    """
    bary = x[:N*2].reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for invalid barycentric coordinates
    penalty = 0.0
    invalid_mask = (u < 0) | (v < 0) | (u + v > 1)
    if np.any(invalid_mask):
        # Quadratic penalty for violation
        viol_u = np.maximum(0, -u)
        viol_v = np.maximum(0, -v)
        viol_w = np.maximum(0, u + v - 1)
        penalty = np.sum(viol_u**2 + viol_v**2 + viol_w**2) * 100.0
        
    # Clamp to valid region for area calculation to avoid NaNs or weird geometry
    bary_clamped = clamp_bary(bary)
    
    areas = calc_areas_bary(bary_clamped)
    min_a = np.min(areas)
    
    return -min_a + penalty

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate diverse initial configurations to seed the global search
    # and to have a fallback.
    inits = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # Random
    for i in range(5):
        inits.append(generate_random_init(seed=1000+i))
        
    # Update best with initial configs
    for init_bary in inits:
        update_best(init_bary)

    # 2. Global Search using Differential Evolution
    # DE is robust for multimodal functions.
    # We define bounds for u and v in [0, 1]. The penalty handles u+v<=1.
    # This explores the square [0,1]x[0,1] for each point, rejecting the corners.
    
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # Strategy: Run DE for a significant portion of the time.
    # DE is stochastic and parallelizable (conceptually), but here it's sequential.
    # We limit maxiter and popsize to fit within time.
    
    # Time budget allocation:
    # 50s for DE, 20s for local polish on best DE result, 10s for fallbacks.
    
    de_start = time.time()
    de_time_limit = 50.0
    
    try:
        # popsize=20, maxiter=100 might be too many. 
        # Let's estimate: N=11, dim=22. 
        # Each eval is ~165 dets. Fast.
        # 20 * 100 * 22 evals = 44000 evals. 
        # 165 dets * 44000 = 7M ops. Should be fast in NumPy.
        
        # Use 'best1bin' crossover, default mutation.
        # tol=1e-8, maxiter=200
        # We will interrupt if time runs out.
        
        # To allow interruption, we can use a custom callback or just run 
        # and check time after. But DE doesn't have a time-based stop directly.
        # We can run in chunks or just hope it finishes. 
        # Given 50s limit and fast evals, it should finish quickly.
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=True,  # Use L-BFGS-B at the end to refine
            init='latinhypercube'
        )
        
        if time.time() < deadline - 20:
            de_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
            update_best(de_bary)
            
    except Exception as e:
        print(f"DE failed: {e}")

    # 3. Local Polish on the best found so far
    # Use Nelder-Mead with a high-beta smooth min to refine the DE result.
    
    if best_bary is not None and time.time() < deadline - 5:
        from scipy.optimize import minimize
        
        def smooth_min_area_bary(bary, beta=100.0):
            areas = calc_areas_bary(bary)
            m = np.min(areas)
            shifted = -beta * (areas - m)
            exp_sum = np.sum(np.exp(shifted))
            if exp_sum < 1e-300:
                return m
            return m - (1.0 / beta) * np.log(exp_sum)

        def nm_objective(x, beta):
            bary = x[:N*2].reshape(N, 2)
            bary = clamp_bary(bary)
            val = smooth_min_area_bary(bary, beta)
            return -val

        # Try a few different beta values for polishing
        current_bary = best_bary.copy()
        current_val = best_val
        
        for beta in [50.0, 200.0, 500.0]:
            if time.time() > deadline - 2:
                break
            
            x0 = current_bary.flatten()
            try:
                res = minimize(
                    nm_objective,
                    x0,
                    args=(beta,),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                refined_bary = clamp_bary(res.x[:N*2].reshape(N, 2))
                refined_val = min_area(refined_bary)
                if refined_val > current_val:
                    current_bary = refined_bary
                    current_val = refined_val
                    update_best(current_bary)
            except Exception:
                pass

    # 4. Final Fallback / Safety
    if best_bary is None:
        best_bary = generate_random_init(42)
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
