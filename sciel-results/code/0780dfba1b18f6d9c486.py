"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Combined strategy: 
- Barycentric coordinates for speed/robustness.
- Stable soft-min objective (max-shift) for high-precision optimization.
- Multi-stage L-BFGS-B with increasing sharpness (beta).
- Nelder-Mead polish on the hard (non-smooth) objective.
"""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute all 165 triangle index combinations
_tri_list = list(combinations(range(N), 3))
_tri_idx = np.array(_tri_list)
_I = _tri_idx[:, 0]
_J = _tri_idx[:, 1]
_K = _tri_idx[:, 2]

def bary_to_cart(bary):
    """Convert barycentric (u, v) to Cartesian (x, y). w = 1-u-v."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coords to valid region: u>=eps, v>=eps, u+v<=1-eps."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums[invalid], 1e-15)
        u[invalid] *= scale
        v[invalid] *= scale
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """Compute areas of all 165 triangles from barycentric coords. Returns array of 165 areas."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]; vi = v[_I]; wi = w[_I]
    uj = u[_J]; vj = v[_J]; wj = w[_J]
    uk = u[_K]; vk = v[_K]; wk = w[_K]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return np.abs(det)

def min_area_bary(bary):
    """Minimum area among all 165 triangles."""
    return float(np.min(calc_areas_bary(bary)))

def soft_min_obj(bary_flat, beta=100.0):
    """
    Stable soft-min objective.
    Uses max-shift to prevent overflow in exp().
    Returns negative soft-min (to be minimized).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # We want to maximize min(areas). 
    # Soft min approximation: -1/beta * log(sum(exp(-beta * areas)))
    # To minimize negative soft min: 1/beta * log(sum(exp(-beta * areas)))
    
    # Let's compute S = sum(exp(-beta * areas))
    # log(S) = log(sum(exp(-beta * areas)))
    # To be stable: let m = min(-beta * areas) = -beta * max(areas)
    # Then log(S) = m + log(sum(exp(-beta * areas - m)))
    
    scaled = -beta * areas
    m = np.max(scaled)
    
    # If m is very large positive, exp(scaled - m) is fine.
    # If m is very large negative, exp(scaled - m) might underflow to 0, but that's ok.
    # Actually, we want to compute log(sum(exp(x))). 
    # Standard trick: max(x) + log(sum(exp(x - max(x))))
    
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    
    soft_min_val = -log_sum / beta
    
    return -soft_min_val

def hard_obj(bary_flat):
    """Hard objective: negative min area."""
    bary = clamp_bary(bary_flat.reshape(N, 2))
    return -min_area_bary(bary)

def generate_hexagonal_init():
    """Generate hexagonal lattice-like initial config in barycentric coords."""
    best_bary = None
    best_val = -np.inf
    for s in np.linspace(0.08, 0.45, 35):
        points = []
        for i in range(8):
            for j in range(8):
                u = i * s
                v = j * s
                if u + v <= 1.0 - 1e-6 and len(points) < N:
                    points.append([u, v])
        if len(points) == N:
            bary = np.array(points)
            val = min_area_bary(bary)
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
    """Random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed(base, noise_scale=0.03, seed=None):
    """Perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_bary = None
    best_score = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        score = min_area_bary(bary)
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
            return True
        return False
    
    def try_optimize(bary_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        if time_left() < 10.0:
            return
        
        try:
            bounds = [(0, 1)] * (N * 2)
            
            # Stage 1: Exploration with smooth landscape
            result1 = minimize(
                soft_min_obj,
                bary_init.flatten(),
                args=(beta1,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            if time_left() < 5.0:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation with sharper landscape
            # We can push beta higher now because soft_min_obj is stable
            result2 = minimize(
                soft_min_obj,
                result1.x,
                args=(beta2,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 15000}
            )
            
            update_best(result2.x.reshape(N, 2))
            
        except Exception:
            pass

    # Phase 1: Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, beta1=50.0, beta2=300.0, maxiter1=300, maxiter2=500)
    
    # Phase 2: Try multiple random initializations
    num_random_starts = 12
    for i in range(num_random_starts):
        if time_left() < 35.0:
            break
        rand_init = generate_random_init(seed=100 + i)
        # Use slightly fewer iterations for random starts to allow more of them
        try_optimize(rand_init, beta1=50.0, beta2=200.0, maxiter1=150, maxiter2=300)
        
    # Phase 3: Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(8):
            if time_left() < 20.0:
                break
            perturbed = generate_perturbed(best_bary, noise_scale=0.02, seed=200 + i)
            # Use very sharp beta for refinement
            try_optimize(perturbed, beta1=100.0, beta2=500.0, maxiter1=200, maxiter2=400)
    
    # Phase 4: Nelder-Mead polish on hard objective
    # Nelder-Mead is derivative-free and good for non-smooth functions
    if best_bary is not None and time_left() > 10.0:
        try:
            res_nm = minimize(
                hard_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            update_best(res_nm.x.reshape(N, 2))
        except Exception:
            pass
            
    # Phase 5: Final L-BFGS-B on hard objective (if time permits)
    # L-BFGS-B might struggle with hard min due to non-differentiability, 
    # but sometimes it can polish if the minimum is unique and sharp.
    if best_bary is not None and time_left() > 5.0:
        try:
            bounds = [(0, 1)] * (N * 2)
            res_final = minimize(
                hard_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxfun': 20000}
            )
            update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
            
    # Fallback
    if best_bary is None:
        best_bary = generate_random_init(seed=0)
    
    # Final safety clamp and convert to Cartesian
    best_bary = clamp_bary(best_bary)
    result_cart = bary_to_cart(best_bary)
    
    # Ensure all points are strictly inside
    result_cart = np.clip(result_cart, 1e-9, None)
    result_cart[:, 1] = np.clip(result_cart[:, 1], 1e-9, H - 1e-9)
    x_min = result_cart[:, 1] / np.sqrt(3)
    x_max = 1.0 - result_cart[:, 1] / np.sqrt(3)
    result_cart[:, 0] = np.clip(result_cart[:, 0], x_min + 1e-9, x_max - 1e-9)
    
    return result_cart
