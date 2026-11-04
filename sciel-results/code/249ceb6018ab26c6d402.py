"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
# Triangle vertices
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])

def _bary_to_cart(bary):
    """Convert barycentric coords (n,3) to Cartesian (n,2)."""
    return bary[:, 0:1] * V[0] + bary[:, 1:2] * V[1] + bary[:, 2:3] * V[2]

def _cart_to_bary(pts):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    n = len(pts)
    bary = np.zeros((n, 3))
    # Solve for barycentric coordinates
    # p = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # Using the fact that V1-V0 and V2-V0 form a basis
    e1 = V[1] - V[0]
    e2 = V[2] - V[0]
    # Invert the 2x2 matrix
    det = e1[0]*e2[1] - e1[1]*e2[0]
    inv_det = 1.0/det
    for i in range(n):
        d = pts[i] - V[0]
        b1 = (d[0]*e2[1] - d[1]*e2[0]) * inv_det
        b2 = (e1[0]*d[1] - e1[1]*d[0]) * inv_det
        b0 = 1.0 - b1 - b2
        bary[i] = [b0, b1, b2]
    return bary

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        a = _triangle_area(points[i], points[j], points[k])
        if a < min_area:
            min_area = a
    return min_area

def _objective(x):
    """
    x = [barycentric coords for N points (3N values), t]
    We minimize -t (to maximize t) subject to all triangle areas >= t
    and barycentric constraints: each point's bary coords sum to 1 and are >= 0
    """
    n_bary = 3 * N
    bary = x[:n_bary].reshape(N, 3)
    t = x[n_bary]
    
    # Normalize barycentric coords (should already sum to 1, but enforce)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    pts = _bary_to_cart(bary)
    
    # Compute all triangle areas
    min_area = float('inf')
    for i, j, k in combinations(range(N), 3):
        a = _triangle_area(pts[i], pts[j], pts[k])
        if a < min_area:
            min_area = a
    
    # We want to maximize t, so minimize -t
    # But we need t <= min_area, so the constraint is t - min_area <= 0
    # For the objective, just return -t
    return -t

def _objective_grad(x):
    """Analytical gradient of the objective."""
    n_bary = 3 * N
    bary = x[:n_bary].reshape(N, 3)
    t = x[n_bary]
    
    bary = bary / bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    
    # Find the minimum area triangle
    min_area = float('inf')
    min_idx = None
    for i, j, k in combinations(range(N), 3):
        a = _triangle_area(pts[i], pts[j], pts[k])
        if a < min_area:
            min_area = a
            min_idx = (i, j, k)
    
    # Gradient of -t with respect to t is -1
    grad = np.zeros_like(x)
    grad[n_bary] = -1.0
    
    # Gradient of min_area with respect to barycentric coords
    # For the triangle (i,j,k), the area is:
    # A = 0.5 * |cross(pj-pi, pk-pi)|
    # The gradient of area w.r.t. each point involves the other two points
    i, j, k = min_idx
    pi, pj, pk = pts[i], pts[j], pts[k]
    
    # cross(pj-pi, pk-pi) = (pjx-pix)(pky-piy) - (pkx-pix)(pjy-piy)
    # dA/dpi = 0.5 * (-(pky-piy), (pkx-pix)) + 0.5 * ((pjy-piy), -(pjx-pix))
    # Actually let me be more careful
    # A = 0.5 * ((pjx-pix)(pky-piy) - (pkx-pix)(pjy-piy))
    # dA/dpix = 0.5 * (-(pky-piy) - (-(pjy-piy))) = 0.5 * (pjy-piy - pky+piy) = 0.5*(pjy-pky)
    # dA/dpiy = 0.5 * (-(pkx-pix) - (-(pjx-pix))) = 0.5*(pjx-pjx... let me redo
    # dA/dpix = 0.5 * (-(pky-piy) + (pjy-piy)) = 0.5*(pjy - pky)
    # dA/dpiy = 0.5 * (-(pkx-pix) + (pjx-pix)) = 0.5*(pjx - pkx)
    # dA/dpjx = 0.5 * (pky-piy)
    # dA/dpjy = 0.5 * (-(pkx-pix)) = 0.5*(pix-pkx)
    # dA/dpkx = 0.5 * (-(pjy-piy)) = 0.5*(piy-pjy)
    # dA/dpky = 0.5 * (pjx-pix)
    
    # But we need to consider the sign (absolute value). For now, assume positive orientation.
    # The gradient of min_area w.r.t. pts[i] is:
    d_pi = np.array([0.5*(pts[j][1]-pts[k][1]), 0.5*(pts[j][0]-pts[k][0])])
    d_pj = np.array([0.5*(pts[k][1]-pts[i][1]), 0.5*(pts[i][0]-pts[k][0])])
    d_pk = np.array([0.5*(pts[i][1]-pts[j][1]), 0.5*(pts[j][0]-pts[i][0])])
    
    # Now convert from Cartesian to barycentric gradient
    # p = b0*V0 + b1*V1 + b2*V2
    # dp/db0 = V0, dp/db1 = V1, dp/db2 = V2
    # So grad_bary = V0 * grad_cart_x_component... no
    # Actually: dp = V0*db0 + V1*db1 + V2*db2
    # So grad_bary[i] = V0 * (dA/dpx) + V1 * (dA/dpy)? No.
    # dA = (dA/dpx)*dpx + (dA/dpy)*dpy
    # dpx = V0x*db0 + V1x*db1 + V2x*db2
    # dpy = V0y*db0 + V1y*db1 + V2y*db2
    # So dA/db0 = dA/dpx * V0x + dA/dpy * V0y
    # dA/db1 = dA/dpx * V1x + dA/dpy * V1y
    # dA/db2 = dA/dpx * V2x + dA/dpy * V2y
    
    def cart_to_bary_grad(d_cart, point_idx):
        return np.array([
            d_cart[0]*V[0][0] + d_cart[1]*V[0][1],
            d_cart[0]*V[1][0] + d_cart[1]*V[1][1],
            d_cart[0]*V[2][0] + d_cart[1]*V[2][1]
        ])
    
    # The constraint is t - min_area <= 0, so the gradient of the constraint w.r.t. x
    # contributes to the KKT conditions. For SLSQP, we provide the objective gradient.
    # The objective is -t, so its gradient is just -1 in the t direction.
    # But we also need the gradient of the constraint for the optimization.
    
    return grad

def _constraints(x):
    """Constraints: t <= min triangle area, and barycentric coords valid."""
    n_bary = 3 * N
    bary = x[:n_bary].reshape(N, 3)
    t = x[n_bary]
    
    bary = bary / bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    
    min_area = _min_triangle_area(pts)
    
    # Constraint: t - min_area <= 0
    return min_area - t

def _constraints_grad(x):
    """Gradient of the constraint function."""
    n_bary = 3 * N
    bary = x[:n_bary].reshape(N, 3)
    t = x[n_bary]
    
    bary = bary / bary.sum(axis=1, keepdims=True)
    pts = _bary_to_cart(bary)
    
    # Find min area triangle
    min_area = float('inf')
    min_idx = None
    for i, j, k in combinations(range(N), 3):
        a = _triangle_area(pts[i], pts[j], pts[k])
        if a < min_area:
            min_area = a
            min_idx = (i, j, k)
    
    i, j, k = min_idx
    d_pi = np.array([0.5*(pts[j][1]-pts[k][1]), 0.5*(pts[j][0]-pts[k][0])])
    d_pj = np.array([0.5*(pts[k][1]-pts[i][1]), 0.5*(pts[i][0]-pts[k][0])])
    d_pk = np.array([0.5*(pts[i][1]-pts[j][1]), 0.5*(pts[j][0]-pts[i][0])])
    
    grad = np.zeros_like(x)
    grad[n_bary] = -1.0  # d(min_area - t)/dt = -1
    
    # d(min_area)/dbary
    def cart_to_bary_grad(d_cart):
        return np.array([
            d_cart[0]*V[0][0] + d_cart[1]*V[0][1],
            d_cart[0]*V[1][0] + d_cart[1]*V[1][1],
            d_cart[0]*V[2][0] + d_cart[1]*V[2][1]
        ])
    
    grad[i*3:(i+1)*3] = cart_to_bary_grad(d_pi)
    grad[j*3:(j+1)*3] = cart_to_bary_grad(d_pj)
    grad[k*3:(k+1)*3] = cart_to_bary_grad(d_pk)
    
    return grad

def _bounds():
    """Bounds for barycentric coordinates (each >= 0) and t."""
    bounds = []
    for i in range(N):
        for j in range(3):
            bounds.append((0.0, 1.0))
    bounds.append((0.0, 1.0))  # t bound
    return bounds

def _initial_barycentric(seed):
    """Generate initial barycentric coordinates using a grid-like pattern with noise."""
    rng = np.random.RandomState(seed)
    bary = np.zeros((N, 3))
    
    # Start with a rough grid and add noise
    # Place points roughly evenly
    angles = rng.uniform(0, 2*np.pi, N)
    radii = rng.uniform(0.1, 0.8, N)
    
    for i in range(N):
        # Random point in triangle using barycentric
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        bary[i] = [1 - r1 - r2, r1, r2]
    
    # Ensure they sum to 1
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_pts = None
    best_score = 0.0
    
    n_bary = 3 * N
    bounds = _bounds()
    
    # Try multiple restarts with different initial configurations
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        bary0 = _initial_barycentric(attempt)
        # Initial t estimate
        pts0 = _bary_to_cart(bary0)
        t0 = _min_triangle_area(pts0) * 0.9
        
        x0 = np.concatenate([bary0.flatten(), [t0]])
        
        # Ensure x0 is feasible
        bary0_norm = bary0 / bary0.sum(axis=1, keepdims=True)
        pts0 = _bary_to_cart(bary0_norm)
        t0 = _min_triangle_area(pts0) * 0.8
        x0 = np.concatenate([bary0_norm.flatten(), [t0]])
        
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                jac=_objective_grad,
                constraints=[{'type': 'ineq', 'fun': _constraints, 'jac': _constraints_grad}],
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < 0:
                bary_opt = result.x[:n_bary].reshape(N, 3)
                bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
                pts_opt = _bary_to_cart(bary_opt)
                
                # Verify points are in triangle
                valid = True
                for p in pts_opt:
                    b = _cart_to_bary(p.reshape(1,2))[0]
                    if np.any(b < -1e-10):
                        valid = False
                        break
                if valid:
                    score = _min_triangle_area(pts_opt)
                    if score > best_score:
                        best_score = score
                        best_pts = pts_opt.copy()
                        
        except Exception:
            continue
    
    # If we didn't find anything good, use a fallback
    if best_pts is None:
        # Use a simple grid-based initialization
        bary = _initial_barycentric(42)
        best_pts = _bary_to_cart(bary)
    
    return best_pts
