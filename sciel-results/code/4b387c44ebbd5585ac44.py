"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Topological Active-Set Pivot - iteratively identify critical triangles,
enforce area constraints via SLSQP, with symmetry-based initialization and local polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools
import math

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
N_TRIP = len(_TRIPLETS)

DEADLINE = 146  # Safety margin from 150s limit

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v).
    x = 0.5 + 0.5v - 0.5u
    y = S3H - S3H*u - S3H*v
    From y: u = (S3H - y) / S3H
    From x: v = 2*(x - 0.5) + u = 2*x - 1 + u
    """
    u = (S3H - pts[:, 1]) / S3H
    v = 2.0 * pts[:, 0] - 1.0 + u
    return np.column_stack([u, v])

def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
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

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    return np.min(_all_triangle_areas(pts))

def _triangle_area_grad(pts, tri_idx):
    """
    Compute gradient of triangle area for a specific triplet index.
    Returns (area, grad) where grad has shape (N, 2).
    """
    i, j, k = _TRIPLETS[tri_idx]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    
    dx1 = pj[0] - pi[0]
    dy1 = pj[1] - pi[1]
    dx2 = pk[0] - pi[0]
    dy2 = pk[1] - pi[1]
    cross = dx1 * dy2 - dy1 * dx2
    area = abs(cross) * 0.5
    
    sign = np.sign(cross)
    if abs(cross) < 1e-15:
        sign = 1.0
    
    grad = np.zeros((N, 2))
    # d(area)/d(pi) = 0.5*sign*(dy1-dy2, dx2-dx1)
    # d(area)/d(pj) = 0.5*sign*(dy2, -dx2)
    # d(area)/d(pk) = 0.5*sign*(-dy1, dx1)
    coeff = 0.5 * sign
    grad[i] = coeff * np.array([dy1 - dy2, dx2 - dx1])
    grad[j] = coeff * np.array([dy2, -dx2])
    grad[k] = coeff * np.array([-dy1, dx1])
    
    return area, grad

def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def _symmetric_init(rng, variant=0):
    """
    Generate symmetric initial configurations.
    The equilateral triangle has 3-fold rotational symmetry and 3 reflection axes.
    """
    bary = np.zeros((N, 2))
    
    if variant == 0:
        # 3 corners + 3 edge midpoints + 1 center + 4 symmetric interior
        bary[0] = [1.0, 0.0]   # V0
        bary[1] = [0.0, 1.0]   # V1
        bary[2] = [0.0, 0.0]   # V2
        bary[3] = [0.5, 0.5]   # Mid V0-V1
        bary[4] = [0.0, 0.5]   # Mid V1-V2
        bary[5] = [0.5, 0.0]   # Mid V2-V0
        bary[6] = [1/3, 1/3]   # Center
        # 4 points in symmetric orbit
        r = 0.15
        bary[7] = [1/3 - r, 1/3 - r]
        bary[8] = [1/3 + r, 1/3 - r]
        bary[9] = [1/3 + r, 1/3 + r]
        bary[10] = [1/3 - r, 1/3 + r]
        
    elif variant == 1:
        # Different symmetric pattern
        bary[0] = [1.0, 0.0]
        bary[1] = [0.0, 1.0]
        bary[2] = [0.0, 0.0]
        bary[3] = [0.4, 0.4]
        bary[4] = [0.0, 0.4]
        bary[5] = [0.4, 0.0]
        bary[6] = [1/3, 1/3]
        bary[7] = [0.2, 0.2]
        bary[8] = [0.5, 0.2]
        bary[9] = [0.2, 0.5]
        bary[10] = [0.35, 0.35]
        
    elif variant == 2:
        # Hexagonal-like pattern
        bary[0] = [1.0, 0.0]
        bary[1] = [0.0, 1.0]
        bary[2] = [0.0, 0.0]
        bary[3] = [1/3, 1/3]
        # Ring of 6 around center
        for i in range(6):
            angle = i * np.pi / 3
            r = 0.2
            u = 1/3 + r * np.cos(angle)
            v = 1/3 + r * np.sin(angle)
            # Project to valid
            if u < 0: u = 0
            if v < 0: v = 0
            if u + v > 1:
                s = u + v
                u /= s
                v /= s
            bary[4+i] = [u, v]
        
    else:
        # Random
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        bary = np.column_stack([u, v])
    
    return _project_bary_to_valid(bary)

def _active_set_optimize(bary0, target_area, max_iter=200):
    """
    Topological Active-Set Pivot:
    1. Find the minimum-area triangle (critical set)
    2. Enforce that triangle's area >= target_area
    3. Use SLSQP with inequality constraints for the critical triangles
    4. Iterate: update critical set, re-optimize
    """
    current_bary = bary0.copy()
    cart = _bary_to_cart(current_bary)
    
    for iteration in range(max_iter):
        # Find critical triangles (those with area close to minimum)
        areas = _all_triangle_areas(cart)
        min_area = np.min(areas)
        
        if min_area >= target_area * 1.001:
            # All triangles above target - success
            break
        
        # Identify active set: triangles with area < target_area * 1.05
        tol = target_area * 0.05
        active_mask = areas < (min_area + tol)
        active_indices = np.where(active_mask)[0]
        
        # Limit to top 15 most critical triangles for constraint management
        if len(active_indices) > 15:
            sorted_idx = active_indices[np.argsort(areas[active_indices])]
            active_indices = sorted_idx[:15]
        
        if len(active_indices) == 0:
            break
        
        # Define SLSQP objective and constraints
        x0 = current_bary.flatten()
        
        def objective(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            c = _bary_to_cart(b)
            a = _all_triangle_areas(c)
            # Maximize minimum area = minimize negative min
            return -np.min(a)
        
        # Inequality constraints: area_i >= target_area for active triangles
        constraints = []
        for idx in active_indices:
            i, j, k = _TRIPLETS[idx]
            def make_constraint(i, j, k):
                def constraint(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    c = _bary_to_cart(b)
                    pi = c[i]
                    pj = c[j]
                    pk = c[k]
                    dx1 = pj[0] - pi[0]
                    dy1 = pj[1] - pi[1]
                    dx2 = pk[0] - pi[0]
                    dy2 = pk[1] - pi[1]
                    cross = dx1 * dy2 - dy1 * dx2
                    area = abs(cross) * 0.5
                    return area - target_area
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j, k)})
        
        # Bounds for barycentric coordinates
        bounds = [(0, 1)] * (N * 2)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-12})
            if res.success or res.fun < objective(x0):
                current_bary = _project_bary_to_valid(res.x.reshape(N, 2))
                cart = _bary_to_cart(current_bary)
        except Exception:
            break
    
    return current_bary

def _nelder_mead_polish(bary0, max_iter=500):
    """Polish with Nelder-Mead on the exact minimum area."""
    x0 = bary0.flatten()
    
    def nm_objective(x):
        b = x.reshape(N, 2)
        b = _project_bary_to_valid(b)
        cart = _bary_to_cart(b)
        return -_min_triangle_area(cart)
    
    try:
        res = minimize(nm_objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        return _project_bary_to_valid(res.x.reshape(N, 2))
    except Exception:
        return bary0

def _gradient_descent_step(bary, target_area, lr=0.01, steps=50):
    """
    Gradient ascent on the minimum area using subgradient of the min.
    For the active set (triangles achieving min), push them up.
    """
    current_bary = bary.copy()
    
    for step in range(steps):
        cart = _bary_to_cart(current_bary)
        areas = _all_triangle_areas(cart)
        min_area = np.min(areas)
        
        # Active set: triangles with area within 1% of min
        tol = min_area * 0.01 + 1e-10
        active_mask = areas <= min_area + tol
        active_indices = np.where(active_mask)[0]
        
        # Compute gradient of min area (subgradient = average of active gradients)
        total_grad = np.zeros((N, 2))
        for idx in active_indices:
            _, g = _triangle_area_grad(cart, idx)
            total_grad += g
        
        total_grad /= max(1, len(active_indices))
        
        # Convert gradient to barycentric space
        # dx/du = -0.5, dx/dv = 0.5
        # dy/du = -S3H, dy/dv = -S3H
        grad_u = total_grad[:, 0] * (-0.5) + total_grad[:, 1] * (-S3H)
        grad_v = total_grad[:, 0] * (0.5) + total_grad[:, 1] * (-S3H)
        bary_grad = np.column_stack([grad_u, grad_v])
        
        # Ascent step
        current_bary = current_bary + lr * bary_grad
        current_bary = _project_bary_to_valid(current_bary)
        
        # Adaptive learning rate
        new_cart = _bary_to_cart(current_bary)
        new_min = _min_triangle_area(new_cart)
        if new_min < min_area:
            lr *= 0.5
        elif new_min > min_area * 1.001:
            lr = min(lr * 1.1, 0.05)
    
    return current_bary

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # Generate diverse initial configurations
    init_configs = []
    
    # Symmetric variants
    for variant in range(3):
        init_configs.append(_symmetric_init(rng, variant))
    
    # Random configurations
    for _ in range(30):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_configs.append(np.column_stack([u, v]))
    
    # Near-boundary configurations
    for _ in range(15):
        u = rng.random(N)
        v = rng.random(N)
        for i in range(N):
            if rng.random() < 0.3:
                edge = rng.integers(0, 3)
                if edge == 0:
                    u[i] = 0
                elif edge == 1:
                    v[i] = 0
                else:
                    s = u[i] + v[i]
                    if s > 0:
                        u[i] /= s
                        v[i] /= s
                    else:
                        u[i] = 0.5
                        v[i] = 0.5
        init_configs.append(np.column_stack([u, v]))
    
    # Evaluate all initial configs
    for bary0 in init_configs:
        if not _elapsed_ok(30):
            break
        _update_best(bary0)
    
    # Main optimization loop
    max_starts = 40
    current_start = 0
    
    for bary0 in init_configs:
        if not _elapsed_ok(25):
            break
        if current_start >= max_starts:
            break
        current_start += 1
        
        current_bary = bary0.copy()
        
        # Phase 1: Gradient descent to increase min area
        if _elapsed_ok(20):
            current_bary = _gradient_descent_step(current_bary, target_area=0.005, lr=0.02, steps=30)
            _update_best(current_bary)
        
        # Phase 2: Active-set SLSQP optimization
        if _elapsed_ok(15):
            # Estimate current min area as target
            cart = _bary_to_cart(current_bary)
            current_min = _min_triangle_area(cart)
            # Set target slightly above current to push improvement
            target = current_min * 1.01 + 0.0001
            current_bary = _active_set_optimize(current_bary, target, max_iter=50)
            _update_best(current_bary)
        
        # Phase 3: Nelder-Mead polish
        if _elapsed_ok(10):
            current_bary = _nelder_mead_polish(current_bary, max_iter=300)
            _update_best(current_bary)
        
        # Phase 4: If time permits, do perturbation + re-optimize
        if _elapsed_ok(15) and current_start < max_starts // 2:
            for _ in range(2):
                if not _elapsed_ok(10):
                    break
                perturbed = current_bary + rng.normal(0, 0.03, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                
                # Quick gradient step
                perturbed = _gradient_descent_step(perturbed, target_area=0.005, lr=0.01, steps=20)
                _update_best(perturbed)
                
                # NM polish
                if _elapsed_ok(8):
                    perturbed = _nelder_mead_polish(perturbed, max_iter=200)
                    _update_best(perturbed)
                    current_bary = perturbed

    # Final polish on the best found
    if best_bary is not None and _elapsed_ok(5):
        # Gradient step with small lr
        if _elapsed_ok(8):
            polished = _gradient_descent_step(best_bary, target_area=best_area, lr=0.005, steps=30)
            _update_best(polished)
        
        # NM polish
        if _elapsed_ok(3):
            polished = _nelder_mead_polish(best_bary, max_iter=800)
            _update_best(polished)
        
        # One more active-set round if time
        if _elapsed_ok(4):
            cart = _bary_to_cart(best_bary)
            current_min = _min_triangle_area(cart)
            target = current_min * 1.005 + 0.00005
            polished = _active_set_optimize(best_bary, target, max_iter=30)
            _update_best(polished)
            if _elapsed_ok(2):
                polished = _nelder_mead_polish(polished, max_iter=300)
                _update_best(polished)

    if best_cart is None:
        # Fallback
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart

# EVOLVE-BLOCK-END
