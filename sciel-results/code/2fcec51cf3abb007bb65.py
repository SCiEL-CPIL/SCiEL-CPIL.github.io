"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0  # seconds before deadline to stop

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    # pts shape (N, 2)
    # indices shape (NUM_TRI, 3)
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _smoothed_obj_and_grad(x, T):
    """
    Compute the smoothed objective (negative log-sum-exp of triangle areas)
    and its gradient.
    x: flattened points (2N,)
    T: temperature parameter
    """
    pts = x.reshape(N, 2)
    
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # To handle the non-differentiability of abs(), we can just use the raw cross product 
    # if we assume the orientation is consistent or just minimize the magnitude.
    # However, the log-sum-exp of abs() is tricky.
    # A common trick is to minimize the sum of exp(-T * area).
    # Let's stick to minimizing -log(sum(exp(-T * area))).
    
    # For numerical stability in log-sum-exp
    max_neg_T_area = np.max(-T * areas)
    exp_terms = np.exp(-T * areas - max_neg_T_area)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(x)
        
    log_sum_exp = max_neg_T_area + np.log(sum_exp)
    obj = -log_sum_exp / T
    
    # Gradient calculation
    # d/dx (-1/T * log(sum exp(-T * A_k)))
    # = -1/T * 1/S * sum (exp(-T * A_k) * (-T * dA_k/dx))
    # = 1/S * sum (exp(-T * A_k) * dA_k/dx)
    
    weights = exp_terms / sum_exp  # Shape (NUM_TRI,)
    
    # Gradient of area A_k = 0.5 * |cross|
    # If we ignore the absolute value sign change (assuming we stay in the same orientation 
    # or that the gradient of the absolute value is handled by the sign), 
    # dA/dx = 0.5 * sign(cross) * d(cross)/dx
    # However, for the smooth surrogate, it's often better to just use the derivative of the 
    # smooth function defined on the positive areas. 
    # Since areas are positive, and we are using exp(-T*area), the gradient of the sum 
    # with respect to x involves d(area)/dx.
    
    # Let's compute d(cross)/dx for each triangle.
    # cross = (x_j - x_i)_x * (y_k - y_i)_y - (y_j - y_i)_y * (x_k - x_i)_x
    # Let's use the vectorized form.
    
    grad_x = np.zeros_like(x)
    
    # We need to accumulate gradients for each point.
    # For a triangle (i, j, k):
    # d(cross)/dx_i = - (dy_j * dy_k - dy_j * dx_k ... wait)
    # Let's expand:
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d/dx_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
    # d/dy_i = -(x_j - x_i) + (x_k - x_i) = x_k - x_j
    # d/dx_j = (y_k - y_i)
    # d/dy_j = -(x_k - x_i)
    # d/dx_k = -(y_j - y_i)
    # d/dy_k = (x_j - x_i)
    
    # Let's compute these for all triangles.
    dx_i = (p_j[:, 1] - p_k[:, 1])
    dy_i = (p_k[:, 0] - p_j[:, 0])
    
    dx_j = (p_k[:, 1] - p_i[:, 1])
    dy_j = -(p_k[:, 0] - p_i[:, 0])
    
    dx_k = -(p_j[:, 1] - p_i[:, 1])
    dy_k = (p_j[:, 0] - p_i[:, 0])
    
    # Weighted sum
    w = weights[:, np.newaxis]
    
    # Accumulate into grad_x
    # Point i
    idx_i = TRI_INDICES[:, 0]
    np.add.at(grad_x, 2 * idx_i, w * dx_i)
    np.add.at(grad_x, 2 * idx_i + 1, w * dy_i)
    
    # Point j
    idx_j = TRI_INDICES[:, 1]
    np.add.at(grad_x, 2 * idx_j, w * dx_j)
    np.add.at(grad_x, 2 * idx_j + 1, w * dy_j)
    
    # Point k
    idx_k = TRI_INDICES[:, 2]
    np.add.at(grad_x, 2 * idx_k, w * dx_k)
    np.add.at(grad_x, 2 * idx_k + 1, w * dy_k)
    
    # Note: The gradient of the absolute value function introduces a sign.
    # If cross < 0, d|cross|/dx = -d(cross)/dx.
    # Our weights are positive. The objective is -log(sum exp(-T * 0.5 * |cross|)).
    # The derivative of -T * 0.5 * |cross| is -0.5 * T * sign(cross) * d(cross)/dx.
    # So the gradient contribution is proportional to sign(cross) * d(cross)/dx.
    # Let's multiply the gradient by the sign of the cross product.
    signs = np.sign(cross)
    # If cross is 0, sign is 0, which is fine (gradient is 0 there? No, it's a cusp).
    # But in practice, it's rare to hit exactly 0.
    
    # We need to apply the sign to the accumulated gradient components?
    # Actually, the gradient of the sum is sum(weights * grad_area).
    # grad_area = 0.5 * sign(cross) * grad_cross.
    # So we should multiply the entire gradient vector by 0.5?
    # The factor 0.5 is inside the area.
    # Let's check the chain rule again.
    # f = -1/T log S, S = sum exp(-T A_k)
    # df/dx = -1/T * 1/S * sum exp(-T A_k) * (-T dA_k/dx) = 1/S sum exp(-T A_k) dA_k/dx
    # dA_k/dx = 0.5 * sign(cross_k) * d(cross_k)/dx
    # So we need to multiply the accumulated gradient (which is sum w * d(cross)/dx) by 0.5 * sign(cross)?
    # Wait, the weights w_k are exp(-T A_k)/S.
    # So grad = sum_k (exp(-T A_k)/S) * 0.5 * sign(cross_k) * d(cross_k)/dx
    # My current grad_x is sum_k w_k * d(cross_k)/dx.
    # So I need to scale it. But the sign is per-triangle.
    # I cannot just multiply the whole vector by a constant sign.
    # I need to re-accumulate with signs included.
    
    # Let's redo the accumulation with signs.
    grad_x = np.zeros_like(x)
    
    # Effective weights including the 0.5 and sign
    # term_k = w_k * 0.5 * sign(cross_k)
    # But wait, if I use abs(cross) in the area, the area is always positive.
    # The derivative of abs(cross) is sign(cross) * d(cross)/dx.
    
    eff_w = weights * 0.5 * signs
    
    np.add.at(grad_x, 2 * idx_i, eff_w * dx_i)
    np.add.at(grad_x, 2 * idx_i + 1, eff_w * dy_i)
    
    np.add.at(grad_x, 2 * idx_j, eff_w * dx_j)
    np.add.at(grad_x, 2 * idx_j + 1, eff_w * dy_j)
    
    np.add.at(grad_x, 2 * idx_k, eff_w * dx_k)
    np.add.at(grad_x, 2 * idx_k + 1, eff_w * dy_k)
    
    return obj, grad_x

def _regular_polygon_init(n, center=(0.5, 0.5), radius=0.4, num_inner=0):
    """Place n points: regular (n-num_inner)-gon around center + num_inner center points."""
    m = n - num_inner
    angles = 2 * np.pi * np.arange(m) / m
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    for _ in range(num_inner):
        pts = np.vstack([pts, np.array(center)])
    return pts

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    if remaining < 5.0:
        return _regular_polygon_init(N)

    best_pts = None
    best_score = -np.inf
    
    # Define starting configurations
    inits = []
    # Regular 13-gon
    inits.append(_regular_polygon_init(N, radius=0.4))
    # Regular 12-gon + center
    inits.append(_regular_polygon_init(N, radius=0.4, num_inner=1))
    # Regular 11-gon + 2 centers (one slightly offset)
    base_11 = _regular_polygon_init(N, radius=0.4, num_inner=2)
    # Perturb the second center
    base_11[12] += np.array([0.1, 0.1])
    inits.append(base_11)
    
    # Add some random perturbations to the regular 13-gon
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _regular_polygon_init(N, radius=0.4)
        perturbed = base + rng.randn(N, 2) * 0.05
        inits.append(perturbed)
        
    # Evaluate initial scores
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
            
    # Optimization loop
    # We will run L-BFGS-B with annealing temperature T
    # T starts small (smooth) and increases (sharp)
    
    for init in inits:
        x = init.flatten()
        
        # Annealing schedule
        # T values: 1.0, 5.0, 10.0, 20.0, 50.0
        # We don't want to spend too much time on each.
        
        for T in [1.0, 5.0, 10.0, 20.0, 50.0]:
            if time.time() - _START > _DEADLINE - _SAFETY:
                break
                
            def fun_and_grad(xk):
                return _smoothed_obj_and_grad(xk, T)
                
            try:
                res = minimize(fun_and_grad, x, method='L-BFGS-B',
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False})
                if res.success or res.status == 0:
                    x = res.x
                    # Evaluate the actual score (not smoothed)
                    pts = x.reshape(N, 2)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
            except Exception:
                pass
                
    return best_pts
# EVOLVE-BLOCK-END
