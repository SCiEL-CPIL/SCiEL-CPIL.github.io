import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def smooth_min_obj(bary_flat, beta=200.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    We maximize the soft min, so we minimize -soft_min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_structured_init():
    """Generate a set of diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal grid points
    def generate_hexagonal_init(scale=0.25):
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
            idx = rng.choice(len(pts), N, replace=False)
            return np.array(pts)[idx].flatten()
        return None

    hex_init = generate_hexagonal_init(0.25)
    if hex_init is not None:
        inits.append(hex_init)
        
    # 2. Random points
    rng = np.random.default_rng(123)
    for _ in range(30):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]).flatten())
        
    # 3. Perturbed hexagonal
    if hex_init is not None:
        for _ in range(5):
            noise = rng.normal(0, 0.05, N * 2)
            perturbed = hex_init + noise
            inits.append(perturbed)
            
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate initial candidates
    inits = generate_structured_init()
    
    # Multi-start gradient descent on smooth objective
    bounds = [(0, 1)] * (N * 2)
    
    # We will run a limited number of starts to stay within time
    # Each L-BFGS-B run is fast (converges in <0.1s usually)
    # We can afford ~50-100 starts.
    
    # Shuffle inits to randomize order, but keep structured ones early? 
    # Actually, mixing is good.
    rng = np.random.default_rng(999)
    rng.shuffle(inits)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Check if this init is already good enough to be the best?
        # Just run the optimizer.
        
        try:
            # Use L-BFGS-B on the smooth objective
            # It handles bounds well.
            result = minimize(
                smooth_min_obj,
                init,
                args=(200.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -hard_min_obj(init):
                update_best(result.x.reshape(N, 2))
                
        except Exception:
            pass
            
    # Local Refinement of the best candidate found so far
    if best_bary is not None and time.time() < deadline - 5:
        # Stage 1: High precision L-BFGS-B on smooth objective with higher beta
        try:
            result1 = minimize(
                smooth_min_obj,
                best_bary.flatten(),
                args=(500.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.success:
                update_best(clamp_bary(result1.x.reshape(N, 2)))
        except Exception:
            pass
        
        # Stage 2: Nelder-Mead on hard objective
        if time.time() < deadline - 2:
            try:
                result2 = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result2.x.reshape(N, 2)))
            except Exception:
                pass

        # Stage 3: SLSQP polish
        if time.time() < deadline - 1:
            try:
                result3 = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result3.x.reshape(N, 2)))
            except Exception:
                pass

    # 3. Final Fallback
    if best_bary is None:
        # Fallback to random if everything failed
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    
    return cart
