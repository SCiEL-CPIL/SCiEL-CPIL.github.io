import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    """Calculate area of triangle formed by three 2D points."""
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])

def _get_areas(points):
    """Calculate all pairwise triangle areas for N points."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _soft_min(x, alpha):
    """
    Compute soft minimum of x using log-sum-exp trick.
    soft_min(x) = -1/alpha * log(sum(exp(-alpha * x)))
    """
    x = np.asarray(x)
    # Shift for numerical stability
    x_min = np.min(x)
    if x_min < -1000: # Handle underflow if values are very negative (though areas are positive)
        return x_min
    # We want to maximize min, so we minimize -min.
    # Let's define the objective as minimizing -soft_min(areas).
    # soft_min(areas) approx min(areas).
    # exp(-alpha * areas)
    # To avoid overflow/underflow, we can use the standard logsumexp.
    # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
    # We want log(sum(exp(-alpha * a_i)))
    # Let z_i = -alpha * a_i.
    # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
    # max(z) = -alpha * min(a)
    # So log(sum(exp(-alpha * a))) = -alpha * min(a) + log(sum(exp(-alpha * (a - min(a)))))
    # soft_min = -1/alpha * logsumexp(-alpha * a)
    #          = min(a) - 1/alpha * log(sum(exp(-alpha * (a - min(a)))))
    
    # Simpler approach: just compute directly if alpha is not huge.
    # But for robustness:
    z = -alpha * x
    z_max = np.max(z)
    # If z_max is very large positive, exp(z - z_max) is safe.
    # z - z_max <= 0.
    exp_sum = np.sum(np.exp(z - z_max))
    log_sum_exp = z_max + np.log(exp_sum)
    return -1.0 / alpha * log_sum_exp

def _objective(params, alpha):
    """
    Objective function to minimize: -soft_min(areas).
    params: array of shape (N, 2)
    """
    points = params.reshape(N, 2)
    areas = _get_areas(points)
    # If any point is invalid (NaN), return large penalty
    if not np.all(np.isfinite(points)):
        return 1e10
    sm = _soft_min(areas, alpha)
    return -sm

def _constraints(params):
    """
    Constraints ensuring points are inside the equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Inequalities g(params) >= 0
    """
    points = params.reshape(N, 2)
    cons = []
    for p in points:
        x, y = p
        # y >= 0
        cons.append(y)
        # y <= sqrt(3) * x  => sqrt(3)*x - y >= 0
        cons.append(np.sqrt(3)*x - y)
        # y <= sqrt(3) * (1-x) => sqrt(3)*(1-x) - y >= 0
        cons.append(np.sqrt(3)*(1-x) - y)
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s buffer for 150s limit
    
    # Triangle vertices
    h = np.sqrt(3) / 2
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, h])
    
    # Initial best
    best_score = -1
    best_points = None
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random points
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            # Map to triangle
            p = v1 + u * (v2 - v1) + v * (v3 - v1)
            pts.append(p)
        candidates.append(np.array(pts).flatten())
        
    # 2. Structured grid-like
    # 3x3 grid points adjusted to fit triangle
    # Rows: y = 0, h/2, h
    # Col counts: 3, 2, 1? No, 11 points.
    # Let's try a specific known good configuration for 10 or 11 points if possible, or just jittered grid.
    # 3x3 grid has 9 points. Add 2 more.
    # Or 4x3 grid?
    # Let's just use a few structured jittered grids.
    for jitter in [0.1, 0.05]:
        pts = []
        # Row 0 (bottom): y=0, x in [0,1]. 4 points.
        for i in range(4):
            x = (i + 0.5)/4.0 + rng.uniform(-jitter, jitter)
            pts.append([x, 0])
        # Row 1 (middle): y=h/2, x in [0.25, 0.75]. 4 points.
        for i in range(4):
            x = 0.25 + (i + 0.5)/4.0 * 0.5 + rng.uniform(-jitter, jitter)
            pts.append([x, h/2])
        # Row 2 (top): y=h, x in [0.5, 0.5]. 1 point.
        pts.append([0.5, h])
        # Total 9. Need 11.
        # Add 2 more in the middle or split bottom.
        # Let's just add 2 random points in the center region.
        for _ in range(2):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            p = v1 + u * (v2 - v1) + v * (v3 - v1)
            pts.append(p)
        candidates.append(np.array(pts).flatten())

    # Optimization loop
    alpha = 50.0  # Softness parameter. Higher is sharper (closer to true min).
    # We might want to anneal alpha, but fixed is simpler and robust if SLSQP is good.
    
    for i, x0 in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Define constraints for SLSQP
        # SLSQP expects constraints as dicts or calls
        # We'll use a function that returns a list of dicts
        def make_constraints(x):
            c = _constraints(x)
            return [{'type': 'ineq', 'fun': lambda x, c=c: c}] # This lambda capture is wrong for SLSQP which calls fun(x)
        
        # Better: define constraint functions properly
        # SLSQP accepts a list of dicts with 'fun' and 'type'.
        # 'fun' should take x and return scalar >= 0.
        
        def cons_fun_factory(idx):
            def fun(x):
                p = x.reshape(N, 2)[idx]
                x_c, y_c = p
                return np.array([
                    y_c,
                    np.sqrt(3)*x_c - y_c,
                    np.sqrt(3)*(1-x_c) - y_c
                ])
            return fun
        
        # SLSQP doesn't take vector constraints directly in a single dict easily unless we split them or use a wrapper.
        # Actually, scipy minimize constraints can be a list of dicts.
        # Each dict 'fun' returns a scalar.
        # So we need 3*N constraint functions.
        
        constraints = []
        for idx in range(N):
            # y >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=idx: x.reshape(N, 2)[idx][1]})
            # sqrt(3)x - y >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=idx: np.sqrt(3)*x.reshape(N, 2)[idx][0] - x.reshape(N, 2)[idx][1]})
            # sqrt(3)(1-x) - y >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=idx: np.sqrt(3)*(1 - x.reshape(N, 2)[idx][0]) - x.reshape(N, 2)[idx][1]})
            
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(alpha,), 
                method='SLSQP', 
                bounds=[(0, 1)] * (2*N), # Basic bounds, constraints are tighter
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            if res.success:
                pts = res.x.reshape(N, 2)
                # Check if points are actually inside (numerical errors)
                valid = True
                for p in pts:
                    x, y = p
                    if y < -1e-6 or np.sqrt(3)*x - y < -1e-6 or np.sqrt(3)*(1-x) - y < -1e-6:
                        valid = False
                        break
                if valid:
                    score = -res.fun # Since we minimized -soft_min
                    # For comparison, use exact min area
                    areas = _get_areas(pts)
                    exact_min = np.min(areas)
                    if exact_min > best_score:
                        best_score = exact_min
                        best_points = pts
                        
        except Exception as e:
            pass

    # If no optimization succeeded, return a valid random set
    if best_points is None:
        best_points = candidates[0].reshape(N, 2)
        
    return best_points
# EVOLVE-BLOCK-END
