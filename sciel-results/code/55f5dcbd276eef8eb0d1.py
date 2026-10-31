import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

def generate_hexagonal_init():
    """Generate a hexagonal/lattice-based initial configuration.
    
    For 11 points in a triangle, a good arrangement places points in a
    triangular lattice pattern. We use 3 rows: 1+3+3+4 or similar.
    Actually for 11 points, let's use a pattern that spreads well.
    
    A known good configuration for maximizing min triangle area uses
    points near the vertices and edges, plus interior points.
    """
    # Use a structured grid that approximates optimal placement
    # Row-based approach: distribute points across horizontal slices
    n = 11
    bary = np.zeros((n, 2))
    
    # Place 3 points near vertices
    bary[0] = [0.15, 0.15]  # near (0,0)
    bary[1] = [0.85, 0.15]  # near (1,0)  
    bary[2] = [0.15, 0.7]   # near (0.5, sqrt(3)/2)
    
    # Place 4 points along edges
    bary[3] = [0.5, 0.05]   # bottom edge middle
    bary[4] = [0.3, 0.4]    # left edge middle
    bary[5] = [0.7, 0.4]    # right edge middle
    bary[6] = [0.5, 0.5]    # center-ish
    
    # Place 4 more interior points
    bary[7] = [0.35, 0.25]
    bary[8] = [0.65, 0.25]
    bary[9] = [0.35, 0.55]
    bary[10] = [0.65, 0.55]
    
    return clamp_bary(bary)

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

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta, more iterations)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min (increasing beta)
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
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
        
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final high-precision Nelder-Mead polish
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

    # 1. Global Search using Differential Evolution
    # DE is robust to the non-smoothness of the min function and explores the space well.
    # We use a soft-min objective with moderate beta to guide DE towards the global optimum
    # without being too noisy, but DE's population-based nature handles the landscape well.
    
    bounds = [(0, 1)] * (N * 2)
    
    # DE callback to check time
    def de_callback(xk, convergence):
        return time.time() > deadline - 60  # Stop DE if we need time for local refinement
    
    # Use a smaller population and fewer generations to fit in time, relying on local refinement
    # popsize=15, maxiter=200
    try:
        de_result = differential_evolution(
            soft_min_obj,
            bounds,
            args=(50.0,),  # Moderate beta for DE
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=False,  # We do our own polishing
            init='sobol', # Better initial distribution
            callback=de_callback
        )
        
        if de_result.success or de_result.fun < -np.inf:
            # DE returns the best point found
            de_bary = clamp_bary(de_result.x.reshape(N, 2))
            update_best(de_bary)
            
            # Also try to refine the top few individuals from the final population if available?
            # SciPy DE doesn't easily expose the final population in all versions via result,
            # but we can just refine the single best found.
            
    except Exception as e:
        pass

    # 2. Local Refinement of the DE result
    if best_bary is not None:
        if time.time() < deadline - 10:
            refined_de = refine_local(best_bary, time_limit=min(40.0, deadline - 10 - time.time()))
            update_best(refined_de)

    # 3. Try structured initializations (hexagonal pattern) if time remains
    if time.time() < deadline - 30:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=min(25.0, deadline - 10 - time.time()))
        update_best(refined_hex)
    
    # 4. Fallback / Additional Search with random starts if time remains
    if time.time() < deadline - 25:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
            rng = np.random.default_rng(seed)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            rand_bary = np.column_stack([u, v])
            
            refined_rand = refine_local(rand_bary, time_limit=min(15.0, deadline - 10 - time.time()))
            update_best(refined_rand)

    # 4. Coordinate Ascent Polish: optimize each point individually
    # This can escape local minima that full-space methods get stuck in
    if best_bary is not None and time.time() < deadline - 10:
        bary_ca = best_bary.copy()
        remaining_time = deadline - 10 - time.time()
        max_sweeps = 3
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 10:
                break
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                # Optimize point i while keeping others fixed
                def point_obj(xi):
                    bary_temp = bary_ca.copy()
                    bary_temp[i] = xi
                    bary_temp = clamp_bary(bary_temp)
                    areas = calc_areas_bary(bary_temp)
                    return -np.min(areas)
                
                try:
                    res_i = minimize(
                        point_obj,
                        bary_ca[i],
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                    )
                    if res_i.fun < -hard_min_obj(bary_ca.flatten()):
                        bary_ca[i] = res_i.x
                        bary_ca = clamp_bary(bary_ca)
                except Exception:
                    pass
        
        update_best(bary_ca)
    
    # 5. Final Polish with very tight Nelder-Mead if time allows
    if best_bary is not None and time.time() < deadline - 5:
        final_polish = refine_local(best_bary, time_limit=min(5.0, deadline - 5 - time.time()))
        update_best(final_polish)

    if best_bary is None:
        # Absolute fallback
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    cart = bary_to_cart(best_bary)
    return cart

if __name__ == "__main__":
    pts = construct()
    print(pts)
