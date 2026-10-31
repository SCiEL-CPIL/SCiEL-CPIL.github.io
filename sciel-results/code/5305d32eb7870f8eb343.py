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
    
    # Gradient of det with respect to u, v for each point
    A = vj * wk - wj * vk
    B = uj * wk - wj * uk
    C = uj * vk - vj * uk
    
    # Position 0 (i)
    g_u_0 = A - B - C
    g_v_0 = -B - C
    
    # Position 1 (j)
    A_j = vi * wk - vk * wi
    B_j = ui * wk - uk * wi
    C_j = ui * vk - vi * uk
    g_u_1 = A_j - B_j - C_j
    g_v_1 = -B_j - C_j
    
    # Position 2 (k)
    A_k = vi * wj - vj * wi
    B_k = ui * wj - uj * wi
    C_k = ui * vj - vi * uj
    g_u_2 = A_k - B_k - C_k
    g_v_2 = -B_k - C_k
    
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    idx_0 = tri_idx[:, 0]
    idx_1 = tri_idx[:, 1]
    idx_2 = tri_idx[:, 2]
    
    sign = np.sign(det)
    
    # Softmax weights
    max_det = np.max(det)
    exp_det = np.exp((det - max_det) / beta)
    sum_exp = np.sum(exp_det)
    weights = exp_det / sum_exp
    
    # Accumulate gradients
    np.add.at(grad_u, idx_0, weights * sign * g_u_0)
    np.add.at(grad_v, idx_0, weights * sign * g_v_0)
    np.add.at(grad_u, idx_1, weights * sign * g_u_1)
    np.add.at(grad_v, idx_1, weights * sign * g_v_1)
    np.add.at(grad_u, idx_2, weights * sign * g_u_2)
    np.add.at(grad_v, idx_2, weights * sign * g_v_2)
    
    final_grad_u = -grad_u / beta
    final_grad_v = -grad_v / beta
    
    grad = np.zeros(2 * N)
    grad[0::2] = final_grad_u
    grad[1::2] = final_grad_v
    
    return grad

def generate_structured_init():
    """
    Generate a structured initialization based on a 3-4-3-1 grid pattern.
    This respects the topology of the optimal packing for 11 points.
    """
    # Rows in barycentric (u, v)
    # Row 0 (bottom): 3 points
    # Row 1: 4 points
    # Row 2: 3 points
    # Row 3 (top): 1 point
    
    # We define approximate positions and then refine
    pts = []
    
    # Bottom row (y=0 in cart, so v=0, w=1-u)
    # Points at x=1/6, 3/6, 5/6 -> v=1/6, 3/6, 5/6, u=1-v
    pts.append([1 - 1/6.0, 1/6.0])
    pts.append([1 - 3/6.0, 3/6.0])
    pts.append([1 - 5/6.0, 5/6.0])
    
    # Second row (y = h/4)
    # In barycentric, y = w * H. So w = 1/4.
    # u + v = 3/4.
    # x = v + 0.5 * w = v + 0.125.
    # We want 4 points roughly evenly spaced.
    # x range in this row: from 0.125 to 0.875 (width 0.75).
    # Spacing 0.75/3 = 0.25.
    # x = 0.125, 0.375, 0.625, 0.875.
    # v = x - 0.125.
    # v = 0, 0.25, 0.5, 0.75.
    # u = 0.75 - v.
    pts.append([0.75 - 0.0, 0.0])
    pts.append([0.75 - 0.25, 0.25])
    pts.append([0.75 - 0.5, 0.5])
    pts.append([0.75 - 0.75, 0.75])
    
    # Third row (y = h/2)
    # w = 1/2. u + v = 1/2.
    # x = v + 0.25.
    # Width 0.5. 3 points.
    # x = 0.25, 0.5, 0.75.
    # v = 0, 0.25, 0.5.
    # u = 0.5 - v.
    pts.append([0.5 - 0.0, 0.0])
    pts.append([0.5 - 0.25, 0.25])
    pts.append([0.5 - 0.5, 0.5])
    
    # Top row (y = h)
    # w = 1. u+v=0.
    # x = 0.5.
    # v=0, u=0.
    pts.append([0.0, 0.0])
    
    bary = np.array(pts)
    return clamp_bary(bary)

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
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if time.time() > start_time + time_limit:
            return bary
            
        candidate_bary = clamp_bary(res.x.reshape(N, 2))
        return candidate_bary
    except Exception:
        return bary

def active_set_refinement(bary, time_limit=10.0):
    """
    Refine only the points involved in the smallest triangles.
    This is a structural change: instead of moving all points, we focus on the "bottleneck" points.
    """
    start_time = time.time()
    current_bary = bary.copy()
    
    # Identify the smallest triangles
    areas = calc_areas_bary(current_bary)
    min_area = np.min(areas)
    
    # Get indices of points in triangles with area close to min
    # We want to move points that are part of small triangles to increase their area
    # Simple heuristic: find the single smallest triangle and move its points
    min_idx = np.argmin(areas)
    tri_pts = tri_idx[min_idx]
    
    # Define an objective that maximizes the area of this specific triangle
    # while keeping other points fixed? No, that's not robust.
    # Instead, let's use a weighted sum where the smallest triangles have high weight.
    
    # We will run a few iterations of this active set refinement
    for _ in range(10):
        if time.time() > start_time + time_limit:
            break
            
        areas = calc_areas_bary(current_bary)
        min_area = np.min(areas)
        
        # Find all triangles with area < min_area * 1.1
        small_tris = np.where(areas < min_area * 1.1)[0]
        
        if len(small_tris) == 0:
            break
            
        # Get unique points involved in these small triangles
        involved_pts = np.unique(tri_idx[small_tris].flatten())
        
        if len(involved_pts) == 0:
            break
            
        # Create a custom objective that penalizes small areas more heavily
        # and only allows movement of involved points?
        # For simplicity, let's just run L-BFGS-B on the full set but with a higher beta
        # to focus on the minimum.
        
        # Actually, a better structural change:
        # Project the involved points away from each other.
        # For each pair of points in involved_pts, if they form a small triangle with a third point,
        # push them apart.
        
        # This is getting complex. Let's stick to a robust L-BFGS-B but with adaptive beta.
        # If the previous run didn't improve, reduce beta to make the objective sharper.
        
        current_val = -hard_min_obj(current_bary.flatten())
        
        # Try refining with a sharper objective
        res = minimize(
            soft_min_obj,
            current_bary.flatten(),
            jac=soft_min_grad,
            method='L-BFGS-B',
            args=(10.0,),  # Sharper beta
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_bary = clamp_bary(res.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        
        if new_val > current_val:
            current_bary = new_bary
        else:
            break
            
    return current_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    # 1. Structured initialization
    bary = generate_structured_init()
    best_bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    # 2. Initial local refinement
    if time.time() < deadline - 20.0:
        refined = refine_local(bary, time_limit=10.0)
        val = -hard_min_obj(refined.flatten())
        if val > best_val:
            best_val = val
            best_bary = refined
            
    # 3. Active set refinement (structural focus on bottlenecks)
    if time.time() < deadline - 10.0:
        active = active_set_refinement(best_bary, time_limit=20.0)
        val = -hard_min_obj(active.flatten())
        if val > best_val:
            best_val = val
            best_bary = active
            
    # 4. Final polish
    if time.time() < deadline - 2.0:
        final = refine_local(best_bary, time_limit=5.0)
        val = -hard_min_obj(final.flatten())
        if val > best_val:
            best_val = val
            best_bary = final
            
    cart = bary_to_cart(best_bary)
    return cart
