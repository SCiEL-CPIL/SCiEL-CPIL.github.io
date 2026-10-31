import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

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

def clamp_bary(bary, eps=1e-10):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    return np.column_stack([u, v])

def hard_min_value(bary):
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = hard_min_value(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False
    
    # Objective for DE: minimize negative min area, with penalty for invalid points
    def de_objective(x_flat):
        bary = x_flat.reshape(N, 2)
        u = bary[:, 0]
        v = bary[:, 1]
        # Check if all points are in valid triangle region
        if np.any(u < -1e-8) or np.any(v < -1e-8) or np.any(u + v > 1 + 1e-8):
            # Penalty proportional to violation
            violation = 0.0
            violation += np.sum(np.maximum(0, -u))
            violation += np.sum(np.maximum(0, -v))
            violation += np.sum(np.maximum(0, u + v - 1))
            return 1.0 + violation
        bary_c = clamp_bary(bary)
        areas = calc_areas_bary(bary_c)
        return -np.min(areas)
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: Differential Evolution - global search
    # Use multiple DE runs with different seeds for robustness
    de_configs = [
        {'maxiter': 800, 'popsize': 35, 'mutation': (0.5, 1.0), 'recombination': 0.8, 'seed': 42},
        {'maxiter': 600, 'popsize': 40, 'mutation': (0.3, 0.9), 'recombination': 0.9, 'seed': 123},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 15:
            break
        try:
            de_result = differential_evolution(
                de_objective,
                bounds=bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                mutation=cfg['mutation'],
                recombination=cfg['recombination'],
                tol=1e-12,
                seed=cfg['seed'],
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < 0:  # Valid solution found
                bary_de = clamp_bary(de_result.x.reshape(N, 2))
                update_best(bary_de)
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement from diverse starting points
    rng = np.random.default_rng(456)
    
    # Build pool of starting points
    start_pool = []
    if best_bary is not None:
        start_pool.append(best_bary.copy())
    
    # Structured starts
    # Centroid
    start_pool.append(np.full((N, 2), 1.0/3.0))
    # Vertices + centroid mix
    verts = np.array([[0,0],[1,0],[0.5, 1.0-H]])  # barycentric: (0,0),(1,0),(0,1)
    for i in range(N):
        # Distribute points among vertices and centroid
        r = rng.random()
        if r < 0.3:
            start_pool.append(verts[i % 3].reshape(1,2).repeat(N, axis=0) + rng.normal(0, 0.1, (N,2)))
        elif r < 0.6:
            start_pool.append(np.full((N,2), 1.0/3.0) + rng.normal(0, 0.15, (N,2)))
        else:
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            start_pool.append(np.column_stack([u, v]))
    
    # Random starts
    for i in range(8):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        start_pool.append(np.column_stack([u, v]))
    
    # Refine each start with Nelder-Mead then SLSQP
    for init_bary in start_pool:
        if time.time() > deadline - 3:
            break
        
        bary_cur = clamp_bary(init_bary.copy())
        
        # Nelder-Mead
        try:
            def nm_obj(x):
                return -hard_min_value(x.reshape(N, 2))
            
            res_nm = minimize(nm_obj, bary_cur.flatten(), method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True})
            bary_cur = clamp_bary(res_nm.x.reshape(N, 2))
        except Exception:
            pass
        
        # SLSQP polish
        if time.time() < deadline - 1:
            try:
                def slsqp_obj(x):
                    return -hard_min_value(x.reshape(N, 2))
                
                res_sls = minimize(slsqp_obj, bary_cur.flatten(), method='SLSQP',
                                  bounds=[(0,1)]*(2*N),
                                  options={'maxiter': 300, 'ftol': 1e-14})
                bary_cur = clamp_bary(res_sls.x.reshape(N, 2))
            except Exception:
                pass
        
        update_best(bary_cur)
    
    # Phase 3: Final precision polish
    if best_bary is not None and time.time() < deadline - 0.5:
        try:
            def final_obj(x):
                return -hard_min_value(x.reshape(N, 2))
            
            res_final = minimize(final_obj, best_bary.flatten(), method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True})
            update_best(clamp_bary(res_final.x.reshape(N, 2)))
        except Exception:
            pass
    
    cart = bary_to_cart(best_bary)
    return cart
