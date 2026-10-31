import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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

def soft_min_obj(bary_flat, beta=50.0):
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
    """
    Generate a structured initial configuration:
    3 vertices, 3 edge midpoints, 3 ring points, 1 center.
    """
    pts = []
    # Vertices (slightly inside to avoid boundary issues in some optimizers, though clamp handles it)
    eps = 0.01
    pts.append([eps, eps])
    pts.append([1-2*eps, eps])
    pts.append([eps, 1-2*eps])
    
    # Edge midpoints
    pts.append([0.5, eps])
    pts.append([0.5, 0.5-eps])
    pts.append([eps, 0.5])
    
    # Center
    pts.append([1/3, 1/3])
    
    # 3 ring points around center
    r = 0.25
    center = [1/3, 1/3]
    for angle in [0, 120, 240]:
        theta = np.radians(angle)
        # Approximate circle in barycentric
        du = r * np.cos(theta)
        dv = r * np.sin(theta) * 0.866
        u = center[0] + du
        v = center[1] + dv
        if u >= 0 and v >= 0 and u + v <= 1:
            pts.append([u, v])
        else:
            # Fallback if outside
            pts.append([center[0], center[1]]) 
            
    # Ensure exactly N points
    while len(pts) < N:
        u = np.random.random()
        v = np.random.random() * (1 - u)
        pts.append([u, v])
        
    return np.array(pts[:N])

def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(10.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.2:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 3: Sharp soft-min
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.6:
        return bary_current

    # Stage 4: Nelder-Mead on Hard Min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final Polish NM
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Phase 1: Global Search with Differential Evolution
    # DE is robust for global optimization.
    # We define the bounds for barycentric coordinates u, v in [0, 1].
    # The constraint u+v <= 1 is handled by clamping in the objective.
    
    # Initial population: Mix of structured and random
    # DE generates its own population, but we can seed it or just let it run.
    # To save time and ensure diversity, we'll run DE for a fixed number of iterations
    # or until time runs out.
    
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # We need to be careful with DE's time.
    # Let's allocate ~70% of the time to DE.
    de_time_budget = TIME_LIMIT * 0.7
    
    # DE parameters
    # popsize: 15 * N * 2 is standard, maybe too big. Let's try 10 * N * 2 = 220.
    # Actually, DE population size is typically 5-15 times the number of dimensions.
    # Dims = 22. Popsize = 15 * 22 = 330.
    # Let's use a smaller popsize to fit in time, e.g., 10 * 22 = 220.
    
    try:
        # maxiter is tricky because we have a time limit.
        # We can use a callback to check time.
        def de_callback(xk, convergence):
            return time.time() > deadline - (TIME_LIMIT * 0.3) # Stop if we need to leave time for refinement
        
        result_de = differential_evolution(
            hard_min_obj,
            bounds,
            maxiter=1000,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False, # We will do our own polish
            callback=de_callback,
            workers=-1 # Use all cores if possible, though DE is sequential in scipy usually? 
                       # Actually scipy DE is single-threaded by default in older versions, 
                       # but newer versions might parallelize. Let's assume single threaded to be safe.
        )
        
        if result_de.success or result_de.fun < best_val:
            update_best(result_de.x.reshape(N, 2))
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback: If DE fails, use structured init
        if best_bary is None:
            update_best(generate_structured_init())
            update_best(generate_random_init(100))

    # Phase 2: Local Refinement on Top Candidates
    # We have best_bary from DE.
    # We might also want to refine a few diverse points if we kept them, 
    # but DE only returns the best.
    # Let's generate a few diverse candidates from the DE result or random perturbations
    # to give the local optimizer a better starting point if DE got stuck in a local min.
    
    candidates_to_refine = [best_bary]
    
    # Add a few perturbed versions of the best
    rng = np.random.default_rng(123)
    for i in range(3):
        noise = rng.normal(0, 0.05, best_bary.shape)
        perturbed = clamp_bary(best_bary + noise)
        candidates_to_refine.append(perturbed)
        
    # Add structured init if it's better than current best (unlikely after DE, but safe)
    struct_init = generate_structured_init()
    val_struct = -hard_min_obj(struct_init.flatten())
    if val_struct > best_val:
        update_best(struct_init)
        candidates_to_refine.append(struct_init)

    # Refine top candidates
    # Sort candidates by value
    candidates_to_refine = [c for c in candidates_to_refine if c is not None]
    if not candidates_to_refine:
        candidates_to_refine = [generate_structured_init()]
        
    def get_val(b):
        return -hard_min_obj(b.flatten())
        
    candidates_to_refine.sort(key=get_val, reverse=True)
    
    # Refine the top 2 distinct candidates
    refined_candidates = []
    for i, cand in enumerate(candidates_to_refine[:2]):
        if time.time() > deadline - 10:
            break
        
        t_limit = min(15.0, deadline - time.time() - 2)
        if t_limit < 5.0:
            break
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            refined_candidates.append(refined)
            update_best(refined)
        except Exception:
            pass

    # Phase 3: Final Polish on the absolute best
    if best_bary is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 1.0:
            try:
                result = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
                )
                update_best(result.x.reshape(N, 2))
            except Exception:
                pass

    if best_bary is None:
        best_bary = generate_structured_init()

    cart = bary_to_cart(best_bary)
    return cart
