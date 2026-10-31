import numpy as np
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
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
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
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

def generate_hexagonal_init():
    pts = []
    scale = 0.3
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

def generate_random_init(seed=123):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v]).flatten()

def generate_perturbed_init(base, seed=456, scale=0.1):
    rng = np.random.default_rng(seed)
    bary = base.reshape(N, 2)
    pert = rng.normal(0, scale, (N, 2))
    new_bary = bary + pert
    return clamp_bary(new_bary).flatten()

def refine_local(bary_flat, deadline):
    # Coordinate Descent on smooth objective
    bary = bary_flat.reshape(N, 2)
    best_val = -hard_min_obj(bary.flatten())
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            for d in range(2):
                for delta in [-0.01, 0.01, -0.001, 0.001, -0.0001, 0.0001]:
                    trial = bary.copy()
                    trial[i, d] += delta
                    trial = clamp_bary(trial)
                    val = -hard_min_obj(trial.flatten())
                    if val > best_val + 1e-12:
                        best_val = val
                        bary = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return bary.flatten(), best_val

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

    # Multi-start local search
    starts = []
    
    hex_init = generate_hexagonal_init()
    if hex_init is not None:
        starts.append(hex_init)
        
    for i in range(20):
        starts.append(generate_random_init(123 + i))
        
    if hex_init is not None:
        for i in range(5):
            starts.append(generate_perturbed_init(hex_init, 456 + i))
            
    for s in starts:
        if time.time() > deadline - 10:
            break
        refined, val = refine_local(s, deadline)
        update_best(refined.reshape(N, 2))
        
    # Final polish with Nelder-Mead on the best candidate
    if best_bary is not None and time.time() < deadline - 2:
        from scipy.optimize import minimize
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            update_best(clamp_bary(result.x.reshape(N, 2)))
        except Exception:
            pass

    if best_bary is None:
        best_bary = np.column_stack([generate_random_init(0).reshape(N, 2)])
        
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
