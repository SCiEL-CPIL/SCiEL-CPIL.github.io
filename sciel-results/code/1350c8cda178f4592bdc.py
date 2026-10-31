import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
TRIS = np.array(list(combinations(range(N), 3)))
TRI_A = TRIS[:, 0]
TRI_B = TRIS[:, 1]
TRI_C = TRIS[:, 2]

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H)
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
    
    ui = u[TRI_A]
    vi = v[TRI_A]
    wi = w[TRI_A]
    
    uj = u[TRI_B]
    vj = v[TRI_B]
    wj = w[TRI_B]
    
    uk = u[TRI_C]
    vk = v[TRI_C]
    wk = w[TRI_C]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-7):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
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
    """
    Objective function using hard min. Returns negative of min area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function using soft min for smoothness.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
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

    def try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Sharp landscape (high beta) to refine the minimum.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration
            def obj1(x):
                return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Check time before stage 2
            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    # Generate diverse initializations
    inits = []
    
    # Hexagonal structured init
    hex_bary = generate_hexagonal_init()
    inits.append(hex_bary)
    
    # Perturbed versions of hexagonal
    for scale in [0.03, 0.08, 0.15]:
        pert = generate_perturbed_init(hex_bary, scale, seed=42)
        inits.append(pert)
    
    # Random inits
    for seed in range(5):
        bary = generate_random_init(seed)
        inits.append(bary)
    
    # Evaluate all inits and sort
    scored_inits = []
    for init in inits:
        val = -hard_min_obj(init.flatten())
        scored_inits.append((init, val))
    
    scored_inits.sort(key=lambda x: x[1], reverse=True)
    
    # Take top candidates and refine
    top_candidates = scored_inits[:5]
    
    for bary_init, init_val in top_candidates:
        if time.time() > deadline - 15:
            break
        
        # Try L-BFGS-B refinement
        try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300)
        
        # Try Nelder-Mead on hard objective
        if time.time() > deadline - 10:
            continue
        
        try:
            def obj_hard(x):
                return hard_min_obj(x)
            
            result_nm = minimize(
                obj_hard,
                bary_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            update_best(result_nm.x.reshape(N, 2))
            
        except Exception:
            pass
    
    # Final high-precision polish if we have a good candidate
    if best_bary is not None and time.time() < deadline - 5:
        try:
            def obj_final(x):
                return soft_min_obj(x, beta=500.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_final = minimize(
                obj_final,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            update_best(result_final.x.reshape(N, 2))
            
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        best_bary = generate_hexagonal_init()
        update_best(best_bary)
    
    # Convert to cartesian for output
    best_cart = bary_to_cart(best_bary)
    
    return best_cart

if __name__ == "__main__":
    pts = construct()
    bary = np.column_stack([pts[:, 0], pts[:, 1]])
    # Convert cart back to bary for scoring
    x = pts[:, 0]
    y = pts[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    bary = np.column_stack([u, v])
    areas = calc_areas_bary(bary)
    print(f"Best score: {np.min(areas)}")
