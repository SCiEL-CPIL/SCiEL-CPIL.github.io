import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

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

def generate_convex_polygon_params():
    """
    Generate parameters for a convex polygon inscribed in the triangle.
    
    We represent 11 points as:
    - 3 corner points (fixed at triangle vertices)
    - 8 points on the edges or interior
    
    Actually, for a convex polygon with 11 vertices, we can parameterize:
    - Angles around the centroid (11 angles, but they must sum to 2π)
    - Radial distances from centroid (11 radii)
    
    But this doesn't guarantee convexity or containment.
    
    Better approach: Use the fact that for Heilbronn's problem, optimal points
    tend to be on or near the boundary. Let's use:
    - 4 points on each edge of the triangle (including corners)
    - That gives 12 points, but corners are shared, so 3 + 4 + 4 + 4 - 3 = 12... 
    - Actually: 3 corners + 3 on each edge (excluding corners) = 3 + 9 = 12
    - We need 11, so: 3 corners + 8 on edges/interior
    
    Let's use a different parameterization:
    - 3 fixed corners
    - 4 points on the bottom edge
    - 2 points on the left edge  
    - 2 points on the right edge
    Total: 3 + 4 + 2 + 2 = 11
    
    Parameters: positions along each edge (8 parameters)
    """
    rng = np.random.default_rng(42)
    
    # Bottom edge: from (0,0) to (1,0) in barycentric: u from 0 to 1, v=0
    # 4 points including endpoints... but corners are separate
    # Let's say 3 points on bottom edge (excluding corners)
    # 2 points on left edge (excluding corners)  
    # 2 points on right edge (excluding corners)
    # Total: 3 + 2 + 2 = 7 interior edge points + 3 corners = 10... need 1 more
    
    # Alternative: 
    # 3 corners
    # 3 on bottom edge (excluding corners)
    # 3 on left edge (excluding corners)
    # 2 on right edge (excluding corners)
    # Total: 3 + 3 + 3 + 2 = 11
    
    params = np.zeros(8)  # 3 + 3 + 2 = 8 edge parameters
    
    # Random initialization
    params[:3] = rng.uniform(0.1, 0.9, 3)  # bottom edge positions
    params[3:6] = rng.uniform(0.1, 0.9, 3)  # left edge positions
    params[6:8] = rng.uniform(0.1, 0.9, 2)  # right edge positions
    
    return params

def edge_params_to_bary(params):
    """
    Convert edge parameters to barycentric coordinates for 11 points.
    
    params: 8 values
    - params[0:3]: positions on bottom edge (u coordinate, v=0)
    - params[3:6]: positions on left edge (v coordinate, u=0)  
    - params[6:8]: positions on right edge (position along edge)
    
    Returns: (11, 2) array of barycentric coordinates
    """
    bary = np.zeros((N, 2))
    
    # 3 corners
    bary[0] = [0.0, 0.0]  # V0
    bary[1] = [1.0, 0.0]  # V1
    bary[2] = [0.0, 1.0]  # V2
    
    # 3 points on bottom edge (from V0 to V1): u varies, v=0
    for i in range(3):
        bary[3 + i] = [params[i], 0.0]
    
    # 3 points on left edge (from V0 to V2): u=0, v varies
    for i in range(3):
        bary[6 + i] = [0.0, params[3 + i]]
    
    # 2 points on right edge (from V1 to V2): 
    # Parametrize: point = (1-t)*V1 + t*V2 in barycentric
    # V1 = (1,0), V2 = (0,1)
    # So point = (1-t, t)
    for i in range(2):
        t = params[6 + i]
        bary[9 + i] = [1.0 - t, t]
    
    return bary

def convex_obj(params):
    """
    Objective for convex polygon optimization.
    """
    bary = edge_params_to_bary(params)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    return -np.min(areas)

def generate_mixed_init():
    """
    Generate a mixed initialization: some points on boundary, some interior.
    
    Strategy: 
    - 3 corners
    - 3 points near bottom edge
    - 3 points near left edge
    - 2 points near right edge
    - With small random perturbations into interior
    """
    rng = np.random.default_rng(42)
    bary = np.zeros((N, 2))
    
    # Corners
    bary[0] = [0.01, 0.01]
    bary[1] = [0.99, 0.01]
    bary[2] = [0.01, 0.99]
    
    # Bottom edge points (with small interior perturbation)
    for i in range(3):
        u = np.linspace(0.2, 0.8, 3)[i]
        bary[3 + i] = [u + rng.normal(0, 0.02), rng.uniform(0.01, 0.05)]
    
    # Left edge points
    for i in range(3):
        v = np.linspace(0.2, 0.8, 3)[i]
        bary[6 + i] = [rng.uniform(0.01, 0.05), v + rng.normal(0, 0.02)]
    
    # Right edge points
    for i in range(2):
        t = np.linspace(0.3, 0.7, 2)[i]
        bary[9 + i] = [(1-t) + rng.normal(0, 0.02), t + rng.normal(0, 0.02)]
    
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with graduated beta
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.6:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.9:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
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

    # 1. Convex polygon optimization (8D parameter space)
    if time.time() < deadline - 60:
        try:
            # Multiple random starts for the convex polygon
            rng = np.random.default_rng(42)
            for trial in range(5):
                if time.time() > deadline - 30:
                    break
                
                # Random initial parameters
                init_params = rng.uniform(0.1, 0.9, 8)
                
                # Differential evolution in 8D
                bounds_edge = [(0.05, 0.95)] * 8
                result_de = differential_evolution(
                    convex_obj,
                    bounds_edge,
                    seed=trial,
                    maxiter=50,
                    popsize=15,
                    tol=1e-6,
                    polish=False
                )
                
                best_params = result_de.x
                bary_de = edge_params_to_bary(best_params)
                bary_de = clamp_bary(bary_de)
                
                # Refine in full 22D space
                refined = refine_local(bary_de, time_limit=min(5.0, deadline - start_time - 20))
                update_best(refined)
        except Exception:
            pass
    
    # 2. Mixed initialization refinement
    if time.time() < deadline - 20:
        try:
            mixed = generate_mixed_init()
            refined_mixed = refine_local(mixed, time_limit=min(10.0, deadline - start_time - 10))
            update_best(refined_mixed)
        except Exception:
            pass
    
    # 3. Basin hopping on best found
    if best_bary is not None and time.time() < deadline - 15:
        bh_time = min(15.0, deadline - start_time - 10.0)
        if bh_time > 3:
            try:
                start_bh = time.time()
                rng = np.random.default_rng(123)
                current_bary = best_bary.copy()
                best_bh_bary = best_bary.copy()
                best_bh_val = best_val
                
                temperature = 0.03
                pert_scale = 0.05
                
                while time.time() - start_bh < bh_time:
                    # Perturb
                    noise = rng.normal(0, pert_scale, current_bary.shape)
                    perturbed = clamp_bary(current_bary + noise)
                    
                    # Local refinement
                    refined = refine_local(perturbed, time_limit=min(2.0, bh_time - (time.time() - start_bh)))
                    refined_val = -hard_min_obj(refined.flatten())
                    
                    # Accept if better or with Metropolis
                    if refined_val > best_bh_val:
                        best_bh_val = refined_val
                        best_bh_bary = refined.copy()
                        current_bary = refined.copy()
                    elif np.log(rng.random()) < (refined_val - best_bh_val) / temperature:
                        current_bary = refined.copy()
                    
                    temperature *= 0.97
                    pert_scale *= 0.99
                    
                    if time.time() - start_bh > bh_time * 0.95:
                        break
                
                update_best(best_bh_bary)
            except Exception:
                pass
    
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass if time permits
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
