"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of the unit equilateral triangle

# Precompute all triangle triples
TRIPLES = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
NTRIP = len(TRIPLES)

# Precompute barycentric coefficient arrays for fast evaluation
# For a triple (i,j,k), area = 0.5 * |det([pj-pi, pk-pi])|
# In terms of flat coords, area = 0.5 * |a*xi + b*yi + c*xj + d*yj + e*xk + f*yk|
# where the coefficients come from the determinant expansion.
# det([pj-pi, pk-pi]) = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
# = xj*yk - xj*yi - xi*yk + xi*yi - yj*xk + yj*xi + yi*xk - yi*xi
# = xj*yk - xi*yk - yj*xk + yi*xk + (yi*xj - yj*xi)  -- wait let me redo
# Actually: (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
# = xj*yk - xj*yi - xi*yk + xi*yi - yj*xk + yj*xi + yi*xk - yi*xi
# = xj*yk - xi*yk - yj*xk + yi*xk + (-xj*yi + yj*xi)
# Group by variables:
# xj: yk
# yj: -xk
# xi: -yk + yj
# yi: xk - xj
# xk: -yj
# yk: xj
# So for each triple, the linear form is:
#   coeff_xi * xi + coeff_yi * yi + coeff_xj * xj + coeff_yj * yj + coeff_xk * xk + coeff_yk * yk
# where (coeff_xi, coeff_yi, coeff_xj, coeff_yj, coeff_xk, coeff_yk) are constants per triple.
# But wait, the coefficients depend on the OTHER points in the triple, so they are NOT constant.
# The area is bilinear/quadratic in the coordinates. Let me just compute directly.

def _min_area_and_grad(flat, beta):
    """Compute smooth min-area surrogate and its gradient.
    flat: array of shape (2N,) with [x0,y0,x1,y1,...,xN-1,yN-1]
    beta: temperature for log-sum-exp
    Returns (value, grad) where value = (1/beta)*log(sum(exp(-beta*area_i)))
    and grad = d(value)/d(flat).
    """
    pts = flat.reshape(N, 2)
    # Compute all triangle areas (2*area = |det|)
    areas = np.empty(NTRIP)
    for idx, (i, j, k) in enumerate(TRIPLES):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        det = dx1 * dy2 - dy1 * dx2
        areas[idx] = 0.5 * abs(det)
    
    # Log-sum-exp for soft-min: f = (1/beta) * log(sum(exp(-beta * area_i)))
    # Use the standard trick for numerical stability
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    value = log_sum / beta
    
    # Gradient: d/dx [ (1/beta) log(sum exp(-beta * area_i)) ]
    # = (1/beta) * sum_i [ exp(-beta*area_i) / sum_j exp(-beta*area_j) * (-beta) * d(area_i)/dx ]
    # = - sum_i [ w_i * d(area_i)/dx ]  where w_i = exp(-beta*area_i) / sum_j exp(-beta*area_j)
    exp_vals = np.exp(neg_beta_areas - max_val)
    w = exp_vals / np.sum(exp_vals)  # weights, sum to 1
    
    grad = np.zeros(2 * N)
    for idx, (i, j, k) in enumerate(TRIPLES):
        wi = w[idx]
        if wi < 1e-15:
            continue
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        det = dx1 * dy2 - dy1 * dx2
        # d(area)/d(coords) where area = 0.5 * |det|
        # For the signed area: s = 0.5 * det
        # ds/dxi = -0.5*(dy2), ds/dyi = 0.5*(dx2)
        # ds/dxj = 0.5*(dy2), ds/dyj = -0.5*(dx2)
        # ds/dxk = -0.5*(dy1), ds/dyk = 0.5*(dx1)
        # But we have abs, so d|s|/d = sign(s) * ds/d
        sign_det = 1.0 if det >= 0 else -1.0
        # d(area)/dxi = 0.5 * sign_det * (-dy2)
        # d(area)/dyi = 0.5 * sign_det * (dx2)
        # d(area)/dxj = 0.5 * sign_det * (dy2)
        # d(area)/dyj = 0.5 * sign_det * (-dx2)
        # d(area)/dxk = 0.5 * sign_det * (-dy1)
        # d(area)/dyk = 0.5 * sign_det * (dx1)
        factor = 0.5 * sign_det * wi
        grad[2 * i + 0] += factor * (-dy2)
        grad[2 * i + 1] += factor * (dx2)
        grad[2 * j + 0] += factor * (dy2)
        grad[2 * j + 1] += factor * (-dx2)
        grad[2 * k + 0] += factor * (-dy1)
        grad[2 * k + 1] += factor * (dx1)
    
    return value, grad


def _true_min_area(pts):
    """Compute the true minimum triangle area."""
    min_a = np.inf
    for (i, j, k) in TRIPLES:
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        a = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
        if a < min_a:
            min_a = a
    return min_a


def _in_triangle(pts, eps=1e-9):
    """Check if all points are inside or on the triangle."""
    for p in pts:
        # Barycentric coordinates
        # p = a*V0 + b*V1 + c*V2, a+b+c=1, a,b,c >= 0
        # Using the standard method:
        # d1 = sign(p, V0, V1), d2 = sign(p, V1, V2), d3 = sign(p, V2, V0)
        x, y = p
        # Using the cross product method
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(p, V[0], V[1])
        d2 = sign(p, V[1], V[2])
        d3 = sign(p, V[2], V[0])
        has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
        has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
        if has_neg and has_pos:
            return False
    return True


def _clamp_to_triangle(pts, margin=0.005):
    """Project points inside the triangle with a small margin."""
    for idx in range(N):
        x, y = pts[idx]
        # Barycentric coordinates
        # a = 1 - x - y/sqrt(3) ... actually for this triangle:
        # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
        # Barycentric: p = a*(0,0) + b*(1,0) + c*(0.5, sqrt(3)/2)
        # x = b + 0.5c, y = c*sqrt(3)/2
        # c = 2y/sqrt(3), b = x - 0.5c = x - y/sqrt(3), a = 1 - b - c
        c = 2 * y / np.sqrt(3)
        b = x - c / 2
        a = 1 - b - c
        # Clamp to [margin, 1-2*margin]
        if a < margin:
            a = margin
        if b < margin:
            b = margin
        if c < margin:
            c = margin
        s = a + b + c
        a /= s
        b /= s
        c /= s
        x_new = b + 0.5 * c
        y_new = c * np.sqrt(3) / 2
        pts[idx, 0] = x_new
        pts[idx, 1] = y_new
    return pts


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    best_flat = None
    best_score = -np.inf
    
    # Initial points: use a grid-like distribution
    # Place points at barycentric coordinates
    initial_configs = []
    
    # Config 1: uniform grid in barycentric coordinates
    # Select 11 well-spread points from the grid
    selected = []
    # Corners (slightly inside)
    selected.append((0.15, 0.15, 0.7))
    selected.append((0.7, 0.15, 0.15))
    selected.append((0.15, 0.7, 0.15))
    # Edge midpoints
    selected.append((0.5, 0.5, 0.0))
    selected.append((0.5, 0.0, 0.5))
    selected.append((0.0, 0.5, 0.5))
    # Center
    selected.append((1/3, 1/3, 1/3))
    # Additional interior points
    selected.append((0.3, 0.3, 0.4))
    selected.append((0.4, 0.3, 0.3))
    selected.append((0.3, 0.4, 0.3))
    
    # Convert selected barycentric coordinates to Cartesian and stack into (N, 2) array
    init_pts_1 = np.zeros((N, 2))
    for idx, bary in enumerate(selected):
        a, b, c = bary
        x = b + 0.5 * c
        y = c * np.sqrt(3) / 2
        init_pts_1[idx] = [x, y]
    initial_configs.append(init_pts_1)
    
    # Also try random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        bary = rng.dirichlet(np.ones(3), size=N)
        pts = np.column_stack([bary[:, 1] + 0.5 * bary[:, 2], bary[:, 2] * np.sqrt(3) / 2])
        initial_configs.append(pts)
    
    # Optimize each initial config
    for init_pts in initial_configs:
        if time.time() > deadline - 5:
            break
        
        init_flat = init_pts.flatten()
        
        # Try multiple beta values (from coarse to fine)
        betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def obj(flat):
                val, _ = _min_area_and_grad(flat, beta)
                return val
            
            def obj_neg(flat):
                val, _ = _min_area_and_grad(flat, beta)
                return -val
            
            def grad_neg(flat):
                _, g = _min_area_and_grad(flat, beta)
                return -g
            
            # Use L-BFGS-B with bounds to keep points roughly in the triangle
            # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
            # But we need to ensure points are inside the triangle
            # We'll use a penalty approach: after each iteration, clamp
            # Actually, let's use the constraint that points are in the triangle
            # via barycentric coordinates, but that's complex for L-BFGS-B
            # Instead, use a large penalty for points outside
            
            def penalized_obj(flat):
                pts = flat.reshape(N, 2)
                val, _ = _min_area_and_grad(flat, beta)
                # Penalty for points outside triangle
                penalty = 0.0
                for p in pts:
                    x, y = p
                    # Check if outside
                    # Line from V0 to V1: y >= 0
                    if y < 0:
                        penalty += (y ** 2) * 100
                    # Line from V1 to V2: x <= 1 - 2y/sqrt(3) ... 
                    # Actually: the line through (1,0) and (0.5, sqrt(3)/2)
                    # slope = (sqrt(3)/2 - 0) / (0.5 - 1) = (sqrt(3)/2) / (-0.5) = -sqrt(3)
                    # y - 0 = -sqrt(3)(x - 1) => y = -sqrt(3)x + sqrt(3)
                    # So y + sqrt(3)*x <= sqrt(3)
                    if y + np.sqrt(3) * x > np.sqrt(3):
                        penalty += ((y + np.sqrt(3) * x - np.sqrt(3)) ** 2) * 100
                    # Line from V2 to V0: x >= 2y/sqrt(3)
                    # Line through (0.5, sqrt(3)/2) and (0,0): slope = (sqrt(3)/2)/0.5 = sqrt(3)
                    # y = sqrt(3)*x => y - sqrt(3)*x <= 0
                    if y - np.sqrt(3) * x > 0:
                        penalty += ((y - np.sqrt(3) * x) ** 2) * 100
                return val + penalty
            
            try:
                result = minimize(
                    penalized_obj,
                    init_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < np.inf:
                    res_pts = result.x.reshape(N, 2)
                    res_pts = _clamp_to_triangle(res_pts)
                    min_a = _true_min_area(res_pts)
                    if _in_triangle(res_pts):
                        if min_a > best_score:
                            best_score = min_a
                            best_flat = res_pts.flatten().copy()
                
                # Update init_flat for next beta
                if result.x is not None:
                    init_flat = result.x
            except Exception:
                pass
    
    if best_flat is None:
        # Fallback: return a simple valid configuration
        pts = initial_configs[0]
        return _clamp_to_triangle(pts)
    
    return best_flat.reshape(N, 2)
