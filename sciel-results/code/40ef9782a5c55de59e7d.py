"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: 
1. Differential Evolution for global topology search.
2. Active Set Identification: Find triangles near the minimum area.
3. Geometric Pivoting / Equalization: Optimize for equal areas among the active set 
   to stabilize the landscape for gradient descent.
4. Soft-Min Annealing: Standard smooth approximation for final polish.
5. Nelder-Mead: Final discrete refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 140  # Safety margin from 150s limit

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    bary: shape (N, 2) where u, v >= 0 and u+v <= 1.
    P = u*V0 + v*V1 + (1-u-v)*V2
    Note: Standard barycentric for vertices V0(0,0), V1(1,0), V2(0.5, S3H)
    x = v*1 + w*0.5 = v + 0.5(1-u-v) = 0.5 + 0.5v - 0.5u
    y = v*0 + w*S3H = S3H(1-u-v) = S3H - S3H*u - S3H*v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = S3H * w
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v) corresponding to V0, V1, V2 weights?
    Actually, let's stick to the (u,v) definition where P = u*V0 + v*V1 + w*V2 is NOT used directly in the previous code's _bary_to_cart.
    The previous code used:
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    With V0=(0,0), V1=(1,0), V2=(0.5, S3H).
    x = v + 0.5w
    y = S3H w
    So w = y/S3H
    v = x - 0.5w
    u = 1 - v - w
    """
    w = pts[:, 1] / S3H
    v = pts[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _all_triangle_areas_cart(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _soft_min_area_and_grad_cart(pts, beta):
    """
    Soft minimum of triangle areas and its gradient w.r.t. pts (Cartesian).
    Returns (soft_min_value, grad) where grad has shape (N, 2).
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = np.log(sum_exp)
    soft_min = max_a - log_sum / beta
    
    w = exp_shifted / sum_exp  # shape (M,)
    
    sign_cross = np.sign(cross)
    sign_cross = np.where(np.abs(cross) < 1e-15, 1.0, sign_cross)
    
    grad = np.zeros((N, 2))
    coeff = w * 0.5 * sign_cross
    
    grad_x_pi = coeff * (dy1 - dy2)
    grad_y_pi = coeff * (dx2 - dx1)
    np.add.at(grad, (_T_I, 0), grad_x_pi)
    np.add.at(grad, (_T_I, 1), grad_y_pi)
    
    grad_x_pj = coeff * dy2
    grad_y_pj = coeff * (-dx2)
    np.add.at(grad, (_T_J, 0), grad_x_pj)
    np.add.at(grad, (_T_J, 1), grad_y_pj)
    
    grad_x_pk = coeff * (-dy1)
    grad_y_pk = coeff * dx1
    np.add.at(grad, (_T_K, 0), grad_x_pk)
    np.add.at(grad, (_T_K, 1), grad_y_pk)
    
    return soft_min, grad

def _cart_to_bary_grad(cart_grad):
    """
    Convert gradient from Cartesian to barycentric coordinates (u, v).
    x = 0.5 + 0.5v - 0.5u
    y = S3H - S3H*u - S3H*v
    
    dx/du = -0.5, dx/dv = 0.5
    dy/du = -S3H, dy/dv = -S3H
    
    grad_u = grad_x * (-0.5) + grad_y * (-S3H)
    grad_v = grad_x * (0.5) + grad_y * (-S3H)
    """
    gu = cart_grad[:, 0] * (-0.5) + cart_grad[:, 1] * (-S3H)
    gv = cart_grad[:, 0] * (0.5) + cart_grad[:, 1] * (-S3H)
    return np.column_stack([gu, gv])

def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates (u,v) are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def _active_set_objective_and_grad(pts, active_indices, beta_mean=100.0, beta_var=10.0):
    """
    Objective that encourages equalization of areas among 'active' triangles.
    We want to maximize the mean area of active triangles while minimizing their variance.
    This creates a smoother landscape around the optimum where multiple triangles share the min area.
    
    Returns (value, grad)
    """
    if len(active_indices) == 0:
        # Fallback to soft min if no active set found
        return _soft_min_area_and_grad_cart(pts, 500.0)
    
    areas = _all_triangle_areas_cart(pts)
    active_areas = areas[active_indices]
    
    mean_a = np.mean(active_areas)
    var_a = np.var(active_areas)
    
    # We want to maximize mean_a and minimize var_a
    # Value = mean_a - lambda * var_a
    # But we also need to ensure these are indeed the minimums? 
    # For the "pivoting" step, we assume the active set is correct.
    # To be safe, we add a penalty if any OTHER triangle is smaller than the mean_active - epsilon.
    min_other = np.min(np.delete(areas, active_indices)) if len(active_indices) < len(areas) else np.inf
    penalty = 0.0
    if min_other < mean_a - 0.001:
        penalty = (mean_a - min_other) ** 2 * 100.0
    
    value = mean_a - beta_var * var_a - penalty
    
    # Gradient calculation
    # d(mean_a)/d(pts) = (1/k) * sum_{i in active} d(a_i)/d(pts)
    # d(var_a)/d(pts) = (2/k) * sum_{i in active} (a_i - mean_a) * d(a_i)/d(pts)
    
    # We reuse the gradient logic from soft min but with specific weights
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    sign_cross = np.sign(cross)
    sign_cross = np.where(np.abs(cross) < 1e-15, 1.0, sign_cross)
    
    # Base gradient of area i w.r.t points is 0.5 * sign * (...)
    # Let G_i be the gradient vector (flattened) for area i.
    # d(value)/d(pts) = sum_{i in active} [ (1/k) - beta_var * (2/k) * (a_i - mean_a) ] * G_i
    
    k = len(active_indices)
    weights = np.zeros(len(areas))
    for idx in active_indices:
        # Coeff for this triangle
        w_i = (1.0 / k) - (2.0 * beta_var / k) * (areas[idx] - mean_a)
        weights[idx] = w_i
    
    coeff = weights * 0.5 * sign_cross
    
    grad = np.zeros((N, 2))
    
    grad_x_pi = coeff * (dy1 - dy2)
    grad_y_pi = coeff * (dx2 - dx1)
    np.add.at(grad, (_T_I, 0), grad_x_pi)
    np.add.at(grad, (_T_I, 1), grad_y_pi)
    
    grad_x_pj = coeff * dy2
    grad_y_pj = coeff * (-dx2)
    np.add.at(grad, (_T_J, 0), grad_x_pj)
    np.add.at(grad, (_T_J, 1), grad_y_pj)
    
    grad_x_pk = coeff * (-dy1)
    grad_y_pk = coeff * dx1
    np.add.at(grad, (_T_K, 0), grad_x_pk)
    np.add.at(grad, (_T_K, 1), grad_y_pk)
    
    return value, grad

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # Initial symmetric configuration
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.2],
        [0.2, 0.4],
    ])
    _update_best(base_sym)

    # Phase 1: Differential Evolution
    def de_objective(flat_bc):
        bc = flat_bc.reshape(-1, 2)
        u = bc[:, 0]
        v = bc[:, 1]
        violation = np.maximum(0, u + v - 1)
        penalty = np.sum(violation ** 2) * 1000.0
        bary = np.column_stack([u, v])
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        return -area + penalty

    bounds_de = [(0, 1), (0, 1)] * N
    
    if _elapsed_ok(60):
        try:
            de_result = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=50,
                popsize=20,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.success or de_result.fun < 1e10:
                de_bary = de_result.x.reshape(N, 2)
                de_bary = _project_bary_to_valid(de_bary)
                _update_best(de_bary)
        except Exception:
            pass

    # Phase 2: Geometric Pivoting / Active Set Equalization
    if best_bary is not None and _elapsed_ok(40):
        current_bary = best_bary.copy()
        
        # Iterate a few times: Identify active set, optimize for equality, repeat
        for pivot_iter in range(3):
            if not _elapsed_ok(10):
                break
            
            cart = _bary_to_cart(current_bary)
            areas = _all_triangle_areas_cart(cart)
            min_a = np.min(areas)
            
            # Identify active triangles: those within 1% of the minimum
            threshold = min_a * 1.01
            active_mask = areas <= threshold
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) < 2:
                # If only 1 or 0 active, fallback to standard soft min for this step
                # Or just break and let phase 3 handle it
                break
                
            def pivot_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                val, cart_grad = _active_set_objective_and_grad(cart, active_indices)
                bary_grad = _cart_to_bary_grad(cart_grad)
                return -val, -bary_grad.flatten()
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(pivot_objective, x0, method='L-BFGS-B', jac=True,
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                if res.success or res.fun < 1e10:
                    current_bary = res.x.reshape(N, 2)
                    current_bary = _project_bary_to_valid(current_bary)
                    _update_best(current_bary)
            except Exception:
                pass

    # Phase 3: Local Polish with Analytic Gradients (L-BFGS-B) - Soft Min Annealing
    if best_bary is not None and _elapsed_ok(40):
        current_bary = best_bary.copy()
        
        betas = [50.0, 200.0, 1000.0, 5000.0, 20000.0]
        
        for beta in betas:
            if not _elapsed_ok(10):
                break
                
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                val, cart_grad = _soft_min_area_and_grad_cart(cart, beta)
                bary_grad = _cart_to_bary_grad(cart_grad)
                return -val, -bary_grad.flatten()
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                if res.success or res.fun < 1e10:
                    current_bary = res.x.reshape(N, 2)
                    current_bary = _project_bary_to_valid(current_bary)
                    _update_best(current_bary)
            except Exception:
                pass

    # Phase 4: Final Polish with Nelder-Mead on Hard Minimum
    if best_bary is not None and _elapsed_ok(15):
        nm_start = best_bary.flatten()
        
        def nm_objective(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res_nm.success:
                nm_bary = res_nm.x.reshape(N, 2)
                nm_bary = _project_bary_to_valid(nm_bary)
                _update_best(nm_bary)
        except Exception:
            pass

    # Fallback
    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart

# EVOLVE-BLOCK-END
