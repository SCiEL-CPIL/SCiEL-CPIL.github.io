import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)
N_TRI = len(tri_indices)

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
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    s = u + v
    bad = s > 1.0 - eps
    if np.any(bad):
        sc = (1.0 - eps) / np.maximum(s, eps)
        u[bad] *= sc[bad]
        v[bad] *= sc[bad]
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])

def hard_min_value(bary):
    areas = calc_areas_bary(bary)
    return float(np.min(areas))

def make_lattice_init():
    """K=3 lattice with perturbed center."""
    pts = []
    for i in range(4):
        for j in range(4-i):
            pts.append([i/3.0, j/3.0])
    # 10 points, perturb the center (1/3,1/3) which is index 5 (i=1,j=1)
    pts[5] = [1/3.0 + 0.04, 1/3.0 + 0.02]
    return np.array(pts)

def make_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def obj_de(x):
    """Objective for Differential Evolution: maximize min area -> minimize -min area"""
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    return -hard_min_value(bary)

def refine_cd(bary_init, time_limit=10.0):
    """
    Coordinate Descent Refinement.
    Iterates over each point, optimizing its (u,v) coordinates while fixing others.
    Uses Nelder-Mead for the 2D subproblem.
    """
    start_time = time.time()
    bary = clamp_bary(bary_init)
    best_val = hard_min_value(bary)
    
    # Pre-compute indices of triangles involving each point for faster sub-evaluation?
    # For now, just recompute all areas in the sub-objective. It's 165 triangles, very fast.
    
    def sub_obj(x_i):
        # x_i is (2,) array of [u, v] for point i
        bary_sub = bary.copy()
        bary_sub[i] = x_i
        bary_sub = clamp_bary(bary_sub)
        return -hard_min_value(bary_sub)
        
    bounds_2d = [(0, 1), (0, 1)]
    
    # Multiple passes to allow convergence
    max_passes = 50
    for p in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            x0 = bary[i]
            try:
                res = minimize(sub_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                new_pt = res.x
                new_pt = clamp_bary(new_pt.reshape(1,2))[0]
                
                bary_new = bary.copy()
                bary_new[i] = new_pt
                bary_new = clamp_bary(bary_new)
                val_new = hard_min_value(bary_new)
                
                if val_new > best_val + 1e-14:
                    bary = bary_new
                    best_val = val_new
                    improved = True
            except Exception:
                pass
        if not improved:
            break
            
    return bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = make_lattice_init()
    best_val = hard_min_value(best_bary)
    
    # 1. Global Search with Differential Evolution
    # DE is robust for finding good basins in high-dim non-smooth problems
    # We limit the time for DE to ~20-30s to leave room for local refinement
    de_time_limit = 25.0
    if time.time() - start_time < de_time_limit:
        try:
            bounds = [(0, 1)] * (2*N)
            # Use a reasonable popsize
            # maxiter is not strictly limited by time in DE, so we check time in callback?
            # Simpler: just run with maxiter and catch timeout? No, DE doesn't support easy timeout callback in standard scipy.minimize wrapper easily without wrapping.
            # We can wrap the objective to check time, but DE evaluates many points.
            # Alternative: Run DE with a fixed number of generations that fits in time.
            # Let's estimate: 22D, popsize 15*22 ~ 330 evals/gen. 
            # If 1 eval is ~0.1ms, 330 evals is 33ms. 100 gens is 3.3s.
            # So we can afford many generations. Let's set maxiter=100 and see.
            
            def de_obj(x):
                if time.time() - start_time > de_time_limit + 1:
                    raise TimeoutError
                return obj_de(x)
                
            res_de = differential_evolution(
                de_obj, bounds, maxiter=100, popsize=20, seed=42, 
                tol=1e-6, polish=False # We will polish ourselves
            )
            if res_de.x is not None:
                bary_de = clamp_bary(res_de.x.reshape(N, 2))
                val_de = hard_min_value(bary_de)
                if val_de > best_val:
                    best_val = val_de
                    best_bary = bary_de
        except TimeoutError:
            pass
        except Exception:
            pass

    # 2. Local Refinement with Coordinate Descent
    # Use remaining time for CD refinement with restarts
    while time.time() - start_time < deadline - 5:
        # Refine current best
        remaining = deadline - time.time() - 2
        if remaining < 2:
            break
        
        # Allocate time for this CD run
        cd_time = min(10.0, remaining * 0.5)
        
        bary_ref, val_ref = refine_cd(best_bary, time_limit=cd_time)
        
        if val_ref > best_val:
            best_val = val_ref
            best_bary = bary_ref
            
        # If we have time, do a random restart perturbation
        if time.time() - start_time < deadline - 10:
            rng = np.random.default_rng(int(time.time() * 1000))
            # Perturb best solution
            noise_scale = 0.05
            noise = rng.normal(0, noise_scale, best_bary.shape)
            bary_pert = clamp_bary(best_bary + noise)
            
            # Refine perturbed solution
            bary_pert_ref, val_pert_ref = refine_cd(bary_pert, time_limit=cd_time)
            if val_pert_ref > best_val:
                best_val = val_pert_ref
                best_bary = bary_pert_ref

    return bary_to_cart(best_bary)
