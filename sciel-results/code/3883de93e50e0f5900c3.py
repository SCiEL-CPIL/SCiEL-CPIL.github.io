# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 140

_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _T_J])
_T_K = np.array([t[2] for t in _T_K])
_N_TRIP = len(_TRIPLETS)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _bary_to_cart(bary):
    """Convert barycentric (a,b,c) with a+b+c=1 to cartesian (x,y)."""
    a = bary[:, 0]
    b = bary[:, 1]
    c = bary[:, 2]
    x = a * V0[0] + b * V1[0] + c * V2[0]
    y = a * V0[1] + b * V1[1] + c * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert cartesian (x,y) to barycentric (a,b,c)."""
    x = cart[:, 0]
    y = cart[:, 1]
    # Solve: x = b*1 + c*0.5, y = c*(sqrt(3)/2)
    c = y / S3H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def _project_bary(bary):
    """Project barycentric coordinates to valid simplex (a,b,c >= 0, sum=1)."""
    # First ensure non-negative
    bary = np.clip(bary, 0.0, None)
    # Normalize to sum to 1
    sums = bary.sum(axis=1, keepdims=True)
    sums[sums < 1e-10] = 1.0
    bary = bary / sums
    return bary

def _min_triangle_area_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    cart = _bary_to_cart(bary)
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas), areas

def _soft_min_area(areas, gamma):
    """Soft minimum using log-sum-exp."""
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - max_a) * gamma)))
    return max_a - np.log(np.sum(np.exp(-(areas - max_a) * gamma))) / gamma

def _objective_lagrangian(bary_flat, multipliers, gamma, penalty):
    """
    Lagrangian objective: minimize -min_area + sum(multipliers * (target_area - area_i)) + penalty * constraint_violation
    """
    bary = _project_bary(bary_flat.reshape(-1, 3))
    cart = _bary_to_cart(bary)
    
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    # Lagrangian term: sum of multipliers * (target - area) for active constraints
    # We use a target area that we're trying to achieve
    target = np.max(areas)  # Use max as a dynamic target
    lagrangian_penalty = np.sum(multipliers * np.maximum(0, target - areas))
    
    # Simplex constraint penalty (should be small since we project)
    a = bary[:, 0]
    b = bary[:, 1]
    c = bary[:, 2]
    simplex_pen = penalty * np.sum(np.maximum(0, -a)**2 + np.maximum(0, -b)**2 + np.maximum(0, -c)**2)
    
    return -min_area + lagrangian_penalty + simplex_pen

def _objective_direct(bary_flat, gamma):
    """Direct objective: maximize soft minimum area."""
    bary = _project_bary(bary_flat.reshape(-1, 3))
    cart = _bary_to_cart(bary)
    
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Soft minimum
    max_a = np.max(areas)
    soft_min = max_a - np.log(np.sum(np.exp(-(areas - max_a) * gamma))) / gamma
    
    return -soft_min

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_area = 0.0

    def _elapsed_ok(margin=10):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area
        bary = _project_bary(bary.copy())
        area, _ = _min_triangle_area_bary(bary)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()

    # Initial candidates in barycentric coordinates
    # Symmetric configuration
    v0_b = np.array([1.0, 0.0, 0.0])
    v1_b = np.array([0.0, 1.0, 0.0])
    v2_b = np.array([0.0, 0.0, 1.0])
    
    m01_b = (v0_b + v1_b) / 2
    m12_b = (v1_b + v2_b) / 2
    m20_b = (v2_b + v0_b) / 2
    c_b = (v0_b + v1_b + v2_b) / 3
    
    init_bary = np.array([
        v0_b, v1_b, v2_b,
        m01_b, m12_b, m20_b,
        c_b,
        (2*v0_b + v1_b + v2_b)/4,
        (v0_b + 2*v1_b + v2_b)/4,
        (v0_b + v1_b + 2*v2_b)/4,
        c_b + np.array([0.05, -0.03, -0.02])
    ])
    
    _update_best(init_bary)
    
    # Lagrangian relaxation approach
    # Dual variables (multipliers) for each triplet constraint
    multipliers = np.zeros(_N_TRIP)
    
    # Subgradient optimization loop
    max_dual_iters = 200
    gamma_schedule = [10.0, 20.0, 50.0, 100.0]
    
    for gamma in gamma_schedule:
        if not _elapsed_ok(15):
            break
            
        for dual_iter in range(max_dual_iters):
            if not _elapsed_ok(5):
                break
            
            # Primal step: minimize Lagrangian using L-BFGS-B
            def primal_obj(x):
                return _objective_lagrangian(x, multipliers, gamma, 100.0)
            
            try:
                res = minimize(primal_obj, best_bary.flatten(), method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                new_bary = _project_bary(res.x.reshape(-1, 3))
                _update_best(new_bary)
            except Exception:
                pass
            
            # Dual step: update multipliers based on constraint violation
            bary = best_bary
            cart = _bary_to_cart(bary)
            pi = cart[_T_I]
            pj = cart[_T_J]
            pk = cart[_T_K]
            dx1 = pj[:, 0] - pi[:, 0]
            dy1 = pj[:, 1] - pi[:, 1]
            dx2 = pk[:, 0] - pi[:, 0]
            dy2 = pk[:, 1] - pi[:, 1]
            cross = dx1 * dy2 - dy1 * dx2
            areas = np.abs(cross) * 0.5
            
            min_area = np.min(areas)
            # Active constraints: those with area close to minimum
            active_mask = areas < (min_area + 0.01 * min_area + 1e-10)
            
            # Subgradient update: increase multipliers for violated/active constraints
            # The gradient of the Lagrangian w.r.t. multipliers is (target - area)
            # We use a step size that decreases over iterations
            step_size = 0.01 / (1.0 + 0.01 * dual_iter)
            
            # Update multipliers: increase for small areas
            multipliers[active_mask] += step_size * (min_area - areas[active_mask])
            multipliers = np.maximum(multipliers, 0.0)  # Keep non-negative
            
            # Occasionally try direct optimization
            if dual_iter % 50 == 0 and _elapsed_ok(10):
                def direct_obj(x):
                    return _objective_direct(x, gamma)
                
                try:
                    res2 = minimize(direct_obj, best_bary.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                    new_bary2 = _project_bary(res2.x.reshape(-1, 3))
                    _update_best(new_bary2)
                except Exception:
                    pass
    
    # Final polish with Nelder-Mead on direct objective
    if best_bary is not None and _elapsed_ok(10):
        def final_obj(x):
            return _objective_direct(x, 50.0)
        
        try:
            res = minimize(final_obj, best_bary.flatten(), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            final_bary = _project_bary(res.x.reshape(-1, 3))
            _update_best(final_bary)
        except Exception:
            pass
    
    # Additional random restarts if time permits
    if best_bary is not None:
        restarts = 0
        while _elapsed_ok(5) and restarts < 5:
            # Generate random barycentric points
            r = rng.random((N, 3))
            r = r / r.sum(axis=1, keepdims=True)
            _update_best(r)
            
            # Optimize from this start
            def restart_obj(x):
                return _objective_direct(x, 30.0)
            
            try:
                res = minimize(restart_obj, r.flatten(), method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                new_bary = _project_bary(res.x.reshape(-1, 3))
                _update_best(new_bary)
            except Exception:
                pass
            
            restarts += 1
    
    if best_bary is None:
        # Fallback: return symmetric configuration
        cart = _bary_to_cart(init_bary)
        return cart
    
    return _bary_to_cart(best_bary)
# EVOLVE-BLOCK-END
