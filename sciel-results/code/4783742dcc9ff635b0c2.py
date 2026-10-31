import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H  # Area of the unit equilateral triangle

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

def clamp_bary(bary, eps=1e-12):
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

def hard_min_obj(x):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(x, beta=50.0):
    """
    Smooth approximation of the max-min objective using log-sum-exp.
    Minimizing -logsumexp(areas/beta) approximates maximizing min(areas).
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # To avoid log(0), add small epsilon
    a = areas + 1e-12
    m = np.max(a)
    return - (m + np.log(np.sum(np.exp((a - m) / beta))) * beta)

def soft_min_grad(x, beta=50.0):
    """
    Gradient of the soft min objective.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Compute areas and their gradients
    # Area_i = |det_i|
    # Since we are maximizing the minimum, and assuming points are in general position,
    # we can treat the determinant sign as fixed locally or use the absolute value gradient.
    # For simplicity and robustness, we compute the gradient of the determinant 
    # and multiply by the sign of the determinant.
    
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
    abs_det = np.abs(det)
    
    # Gradient of det with respect to u, v for each point
    # det = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_k)
    # Note: w_i = 1 - u_i - v_i
    
    # d(det)/du_i = (v_j w_k - v_k w_j) - (u_j w_k - u_k w_j) - (u_j v_k - u_k v_k)
    # d(det)/dv_i = (v_j w_k - v_k w_j) - (u_j w_k - u_k w_j) - (u_j v_k - u_k v_k) ? No.
    # Let's derive carefully.
    # det = u_i A - v_i B + (1-u_i-v_i) C
    # where A = v_j w_k - v_k w_j, B = u_j w_k - u_k w_j, C = u_j v_k - u_k v_k
    # d(det)/du_i = A - B - C
    # d(det)/dv_i = -B - C
    
    # Wait, C depends on u_j, v_j, u_k, v_k. It does NOT depend on u_i, v_i.
    # So for point i:
    # d(det)/du_i = A - B - C
    # d(det)/dv_i = -B - C
    
    # Let's compute these terms for all triangles.
    A = vj * wk - vk * wj
    B = uj * wk - uk * wj
    C = uj * vk - uk * vk  # Wait, C = u_j v_k - u_k v_k? No.
    # C = u_j v_k - u_k v_k is wrong.
    # C = u_j v_k - u_k v_k?
    # Original: w_i (u_j v_k - u_k v_k) ?? No.
    # Original term: w_i (u_j v_k - u_k v_k) is incorrect expansion.
    # det = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_k)
    # Let's re-verify the determinant expansion.
    # | u_i v_i w_i |
    # | u_j v_j w_j |
    # | u_k v_k w_k |
    # = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    # So A = v_j w_k - w_j v_k
    #    B = u_j w_k - w_j u_k
    #    C = u_j v_k - v_j u_k
    
    A = vj * wk - wj * vk
    B = uj * wk - wj * uk
    C = uj * vk - vj * uk
    
    # Gradient for point i in triangle t:
    # d(det_t)/du_i = A_t - B_t - C_t
    # d(det_t)/dv_i = -B_t - C_t
    
    grad_u_i = A - B - C
    grad_v_i = -B - C
    
    # These are gradients of the determinant for the triangle where i is the first index.
    # We need to accumulate gradients for all points in all triangles.
    
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    # For each triangle t, and for each point i in t:
    # If i is index 0: d(det)/du_i = A-B-C, d(det)/dv_i = -B-C
    # If i is index 1: d(det)/du_i = ... (symmetry)
    # If i is index 2: d(det)/du_i = ... (symmetry)
    
    # Due to symmetry of the determinant, the gradient structure is similar.
    # Let's compute gradients for all 3 positions.
    
    # Position 0 (i):
    # d(det)/du_i = A - B - C
    # d(det)/dv_i = -B - C
    
    # Position 1 (j):
    # Swap i and j in the formula?
    # det = u_j(v_i w_k - v_k w_i) - v_j(u_i w_k - u_k w_i) + w_j(u_i v_k - v_i u_k)
    # A' = v_i w_k - v_k w_i
    # B' = u_i w_k - u_k w_i
    # C' = u_i v_k - v_i u_k
    # d(det)/du_j = A' - B' - C'
    # d(det)/dv_j = -B' - C'
    
    # Position 2 (k):
    # A'' = v_i w_j - v_j w_i
    # B'' = u_i w_j - u_j w_i
    # C'' = u_i v_j - v_i u_j
    # d(det)/du_k = A'' - B'' - C''
    # d(det)/dv_k = -B'' - C''
    
    # Compute A, B, C for all permutations efficiently?
    # It's easier to just loop over triangles or use vectorized ops for each position.
    
    # Let's use vectorized ops for position 0, 1, 2.
    
    # Pos 0 (index 0 of tri_idx)
    # A0, B0, C0 are already computed above as A, B, C.
    g_u_0 = A - B - C
    g_v_0 = -B - C
    
    # Pos 1 (index 1 of tri_idx)
    # We need A1, B1, C1 based on indices (1, 0, 2) effectively?
    # Let's just recompute with the correct indices.
    # For point at index 1 (j), the "other" points are i (0) and k (2).
    # u_i, v_i, w_i are u[tri_idx[:,0]], etc.
    # u_j, v_j, w_j are u[tri_idx[:,1]], etc.
    # u_k, v_k, w_k are u[tri_idx[:,2]], etc.
    
    # For j:
    # A_j = v_i w_k - v_k w_i
    # B_j = u_i w_k - u_k w_i
    # C_j = u_i v_k - v_i u_k
    
    A_j = vi * wk - vk * wi
    B_j = ui * wk - uk * wi
    C_j = ui * vk - vi * uk
    
    g_u_1 = A_j - B_j - C_j
    g_v_1 = -B_j - C_j
    
    # For k:
    # A_k = v_i w_j - v_j w_i
    # B_k = u_i w_j - u_j w_i
    # C_k = u_i v_j - v_i u_j
    
    A_k = vi * wj - vj * wi
    B_k = ui * wj - uj * wi
    C_k = ui * vj - vi * uj
    
    g_u_2 = A_k - B_k - C_k
    g_v_2 = -B_k - C_k
    
    # Now accumulate into grad_u, grad_v
    # We need to map back to the original point indices.
    idx_0 = tri_idx[:, 0]
    idx_1 = tri_idx[:, 1]
    idx_2 = tri_idx[:, 2]
    
    # Sign of determinant
    sign = np.sign(det)
    
    # Weight for soft min
    # p_t = exp((det_t - max_det) / beta) / sum(exp(...))
    # Actually, the gradient of -logsumexp(a/beta) is -1/beta * softmax(a/beta) * grad(a)
    # Let a_t = det_t.
    # grad_obj = -1/beta * softmax(det/beta) * grad_det
    # But we want to maximize min, so we minimize -min.
    # Soft min approx: -logsumexp(det/beta).
    # grad(-logsumexp(det/beta)) = -1/beta * softmax(det/beta) * grad_det
    
    # Calculate softmax weights
    max_det = np.max(det)
    exp_det = np.exp((det - max_det) / beta)
    sum_exp = np.sum(exp_det)
    weights = exp_det / sum_exp  # This is softmax(det/beta)
    
    # Gradient of det with respect to coordinates
    # We have g_u_0, g_v_0 etc. which are grad(det)/du_i for the specific point.
    # We need to multiply by sign(det) because area = |det|.
    # If det > 0, d(area)/du = d(det)/du.
    # If det < 0, d(area)/du = -d(det)/du.
    # So d(area)/du = sign(det) * d(det)/du.
    
    g_area_u_0 = sign * g_u_0
    g_area_v_0 = sign * g_v_0
    g_area_u_1 = sign * g_u_1
    g_area_v_1 = sign * g_v_1
    g_area_u_2 = sign * g_u_2
    g_area_v_2 = sign * g_v_2
    
    # Accumulate
    np.add.at(grad_u, idx_0, weights * g_area_u_0)
    np.add.at(grad_v, idx_0, weights * g_area_v_0)
    np.add.at(grad_u, idx_1, weights * g_area_u_1)
    np.add.at(grad_v, idx_1, weights * g_area_v_1)
    np.add.at(grad_u, idx_2, weights * g_area_u_2)
    np.add.at(grad_v, idx_2, weights * g_area_v_2)
    
    # The objective is -logsumexp(det/beta).
    # grad_obj = -1/beta * (accumulated gradient of det weighted by softmax)
    # Wait, the accumulation above is sum_t weights_t * grad_det_t.
    # So grad_obj = -1/beta * [grad_u, grad_v]
    
    final_grad_u = -grad_u / beta
    final_grad_v = -grad_v / beta
    
    grad = np.zeros(2 * N)
    grad[0::2] = final_grad_u
    grad[1::2] = final_grad_v
    
    return grad

def generate_hexagonal_init():
    """
    Generate a hexagonal grid initialization.
    """
    pts = []
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    
    pts.append([1/3.0, 1/3.0])
    
    bary = np.array(pts)
    return clamp_bary(bary)

def generate_random_init(rng):
    """Generate a random valid configuration."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def refine_local(bary, time_limit=5.0):
    """
    Perform a local refinement using L-BFGS-B on the soft min objective.
    """
    start_time = time.time()
    
    try:
        res = minimize(
            soft_min_obj,
            bary.flatten(),
            jac=soft_min_grad,
            method='L-BFGS-B',
            args=(50.0,),
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > start_time + time_limit:
            return bary
            
        candidate_bary = clamp_bary(res.x.reshape(N, 2))
        return candidate_bary
    except Exception:
        return bary

def annealed_coordinate_descent(initial_bary, start_temp=0.05, end_temp=1e-6, time_limit=30.0):
    """
    Perform Annealed Coordinate Descent.
    This is more efficient than full SA because it exploits the low-dimensional nature
    of moving single points.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    current_bary = initial_bary.copy()
    current_val = -hard_min_obj(current_bary.flatten())
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    while time.time() - start_time < time_limit:
        # Cooling schedule
        progress = min(1.0, (time.time() - start_time) / time_limit)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Pick a random point to move
        idx = rng.integers(0, N)
        
        # Generate a new position for this point
        # Perturb in barycentric coords
        new_point = current_bary[idx].copy()
        scale = 0.1 * (1.0 + temp)
        new_point += rng.normal(0, scale, size=2)
        
        new_bary = current_bary.copy()
        new_bary[idx] = new_point
        new_bary = clamp_bary(new_bary)
        
        new_val = -hard_min_obj(new_bary.flatten())
        
        # Acceptance
        delta = new_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-9)):
            current_bary = new_bary
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
                
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    rng = np.random.default_rng(42)
    
    # 1. Initial diverse starts
    inits = [generate_hexagonal_init()]
    for _ in range(10):
        inits.append(generate_random_init(rng))
    
    # Sort by initial quality
    init_scores = []
    for bary in inits:
        val = -hard_min_obj(bary.flatten())
        init_scores.append((val, bary))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Refine top initial points with L-BFGS-B (soft min)
    for val, bary in init_scores:
        if time.time() > deadline - 30.0:
            break
        polished = refine_local(bary, time_limit=2.0)
        p_val = -hard_min_obj(polished.flatten())
        if p_val > best_val:
            best_val = p_val
            best_bary = polished
            
    # 3. Annealed Coordinate Descent from the best solution
    if best_bary is not None:
        if time.time() < deadline - 10.0:
            acd_time = min(40.0, deadline - time.time() - 5.0)
            acd_bary = annealed_coordinate_descent(best_bary, start_temp=0.05, end_temp=1e-5, time_limit=acd_time)
            acd_val = -hard_min_obj(acd_bary.flatten())
            if acd_val > best_val:
                best_val = acd_val
                best_bary = acd_bary
                
    # 4. Final local refinement
    if best_bary is not None:
        if time.time() < deadline - 2.0:
            final_bary = refine_local(best_bary, time_limit=3.0)
            final_val = -hard_min_obj(final_bary.flatten())
            if final_val > best_val:
                best_val = final_val
                best_bary = final_bary
                
    if best_bary is None:
        best_bary = inits[0]
        
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
