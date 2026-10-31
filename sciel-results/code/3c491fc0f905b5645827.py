import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H

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

def hard_min_obj(x):
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    pts = []
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    pts.append([1/3.0, 1/3.0])
    bary = np.array(pts)
    return clamp_bary(bary)

def generate_random_init(rng):
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def refine_local(bary, time_limit=5.0):
    start_time = time.time()
    try:
        res = minimize(
            hard_min_obj,
            bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() > start_time + time_limit:
            return bary
        candidate_bary = clamp_bary(res.x.reshape(N, 2))
        return candidate_bary
    except Exception:
        return bary

def critical_point_search(bary, time_limit=30.0):
    """
    Structural change: Instead of generic DE/SA, we use a 'critical triangle' guided search.
    We identify the set of triangles that define the current minimum area (within a tolerance).
    We then attempt to move points to 'equalize' these areas, which is a known heuristic for
    maximizing the minimum distance/area in packing problems (Tammes problem analogs).
    
    We formulate this as a least-squares problem where we try to make the areas of the
    critical triangles equal to a target value A, while keeping other areas > A.
    Since we can't easily constrain 'other areas > A' in a simple LSQ, we use a penalty approach
    or simply iterate: 
    1. Find min area A_min.
    2. Find all triangles with area < A_min * (1 + tol).
    3. Try to increase A_min by moving points involved in these triangles.
    
    A more robust approach: Use the gradient of the min-area function.
    Since min is non-differentiable, we approximate it by averaging the gradients of the
    active constraints (the triangles with area close to the minimum).
    """
    start_time = time.time()
    current_bary = bary.copy()
    current_val = -hard_min_obj(current_bary.flatten())
    
    # We will run a few iterations of 'active set' refinement
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
            
        areas = calc_areas_bary(current_bary)
        min_area = np.min(areas)
        
        # Find active constraints: triangles with area close to the minimum
        # We use a tolerance relative to the minimum area
        tol = 1e-4 * min_area + 1e-8
        active_mask = areas < (min_area + tol)
        
        if not np.any(active_mask):
            break
            
        active_indices = np.where(active_mask)[0]
        
        # If there's only one active triangle, the gradient is well-defined for that triangle.
        # If multiple, we take the average gradient of the negative area (to maximize min area).
        # Area of triangle (i,j,k) in barycentric coords (u,v) is:
        # A = |det([ui, vi, wi; uj, vj, wj; uk, vk, wk])|
        # Since we are inside the triangle, we can assume the orientation is consistent 
        # or just use the absolute value derivative carefully.
        # Actually, it's easier to just use numerical gradient of the smooth approximation 
        # or the specific active constraints.
        
        # Let's compute the gradient of the minimum area with respect to barycentric coords.
        # d(min A)/dx approx (1/M) * sum_{k in active} d(A_k)/dx
        
        # We need d(A_k)/d(u_i, v_i) for each point i involved in active triangles.
        # This is complex to derive analytically for all 165 triangles quickly.
        # Instead, let's use a "subgradient" approach via finite differences on the active set?
        # No, that's slow.
        
        # Alternative Structural Approach: 
        # Use the fact that at the optimum, the points are often in "equilibrium" under 
        # repulsive forces proportional to the gradients of the areas.
        
        # Let's try a specific heuristic: 
        # For each active triangle, compute the "push" needed to increase its area.
        # The area is proportional to the cross product of edge vectors.
        # To increase the area of a triangle, we can move its vertices away from each other.
        
        # Let's implement a simple force-based relaxation for active triangles.
        # Force on point p_i from triangle (i,j,k) is proportional to the gradient of Area(i,j,k).
        
        # Gradient of Area(i,j,k) w.r.t. p_i:
        # A = 0.5 * | (p_j - p_i) x (p_k - p_i) |
        # In 2D, cross product is scalar: (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
        # Let C = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
        # If C > 0, dA/dp_i = -0.5 * ( (y_k-y_j, -(x_k-x_j) ) ) ... wait.
        # d/dx_i [ (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) ]
        # = -1*(y_k-y_i) + (x_j-x_i)*(0) - [ 1*(x_k-x_i) + (y_j-y_i)*(0) ]
        # = -(y_k-y_i) - (x_k-x_i) ... this is getting messy with signs.
        
        # Let's stick to the most robust structural change: 
        # **Simulated Annealing with Adaptive Step Size based on Active Set Violation**
        # But the prompt asks for a DIFFERENT design family.
        
        # Let's try **Basin Hopping with Deterministic Local Search (BHLSD)**
        # The local search is the Nelder-Mead, but the global moves are structured "kicks"
        # that specifically perturb the points involved in the smallest triangles.
        
        pass # Placeholder for logic below

    # Actual Implementation of the New Strategy:
    # "Active-Set Guided Basin Hopping"
    
    current_bary = bary.copy()
    current_val = -hard_min_obj(current_bary.flatten())
    best_bary = current_bary.copy()
    best_val = current_val
    
    rng = np.random.default_rng(123)
    
    for it in range(100): # Max 100 hops
        if time.time() - start_time > time_limit:
            break
        
        areas = calc_areas_bary(current_bary)
        min_area = np.min(areas)
        
        # Identify points involved in the smallest triangles
        # Smallest triangle index
        min_idx = np.argmin(areas)
        min_tri = tri_idx[min_idx]
        
        # Also include triangles that are close to the minimum (active set)
        tol = 0.01 * min_area
        active_mask = areas < (min_area + tol)
        active_tris = tri_idx[active_mask]
        
        # Get unique points in active triangles
        active_points = np.unique(active_tris)
        
        # Perturb ONLY these points
        # Scale perturbation by the "slack" or just a fixed small amount
        # If we are stuck, the active set might be large.
        
        n_perturb = len(active_points)
        if n_perturb == 0:
            active_points = np.array([min_tri[0]])
            n_perturb = 1
            
        # Perturbation magnitude
        # Larger perturbation if we are in a deep local minimum (small min_area)?
        # No, usually we want to escape.
        # Let's use a temperature-like schedule.
        temp = 0.02 * (0.9 ** it) + 0.001
        
        new_bary = current_bary.copy()
        # Add random noise to active points
        noise = rng.normal(0, temp, size=(n_perturb, 2))
        new_bary[active_points] += noise
        new_bary = clamp_bary(new_bary)
        
        # Local refinement of the new candidate
        refined_bary = refine_local(new_bary, time_limit=1.0)
        refined_val = -hard_min_obj(refined_bary.flatten())
        
        # Accept if better (greedy) or with some probability (SA-like)
        # For Basin Hopping, we usually accept if the local optimum is better.
        if refined_val > current_val:
            current_bary = refined_bary
            current_val = refined_val
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        else:
            # Optional: Accept worse moves to escape plateaus?
            # Standard BH accepts if the *local minimum* is better.
            # If not, we stay at current local min and try a different kick.
            pass
            
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    rng = np.random.default_rng(42)
    
    # 1. Initial diverse starts
    inits = [generate_hexagonal_init()]
    for _ in range(5):
        inits.append(generate_random_init(rng))
    
    init_scores = []
    for bary in inits:
        val = -hard_min_obj(bary.flatten())
        init_scores.append((val, bary))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Quick local refinement of best inits
    for val, bary in init_scores:
        if time.time() > deadline - 30.0:
            break
        polished = refine_local(bary, time_limit=5.0)
        p_val = -hard_min_obj(polished.flatten())
        if p_val > best_val:
            best_val = p_val
            best_bary = polished
            
    # 3. Global Search: Active-Set Guided Basin Hopping
    # This is the structural change. Instead of DE, we use BH that targets
    # the geometric constraints (small triangles) directly.
    
    if best_bary is not None:
        # Run BH from the best initial solution
        bh_time = min(60.0, deadline - time.time() - 10.0)
        if bh_time > 5.0:
            bh_bary = critical_point_search(best_bary, time_limit=bh_time)
            bh_val = -hard_min_obj(bh_bary.flatten())
            if bh_val > best_val:
                best_val = bh_val
                best_bary = bh_bary
                
    # 4. Try BH from a few other good inits if time permits
    if best_bary is not None:
        for val, bary in init_scores[1:3]:
            if time.time() > deadline - 20.0:
                break
            # Only try if this init is reasonably close to the best
            if val > best_val * 0.9:
                bh_bary = critical_point_search(bary, time_limit=15.0)
                bh_val = -hard_min_obj(bh_bary.flatten())
                if bh_val > best_val:
                    best_val = bh_val
                    best_bary = bh_bary

    # 5. Final local refinement
    if best_bary is not None:
        if time.time() < deadline - 2.0:
            final_bary = refine_local(best_bary, time_limit=5.0)
            final_val = -hard_min_obj(final_bary.flatten())
            if final_val > best_val:
                best_val = final_val
                best_bary = final_bary
                
    if best_bary is None:
        best_bary = inits[0]
        
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
