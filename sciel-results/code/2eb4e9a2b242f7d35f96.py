"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute minimum area among all C(n,3) triangles."""
    n = len(points)
    min_area = float('inf')
    min_triple = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
                    min_triple = (i, j, k)
    return min_area, min_triple

def _inside_triangle(p, tol=1e-9):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p
    if y < -tol or y > h + tol:
        return False
    if x < y / np.sqrt(3) - tol:
        return False
    if x > 1 - y / np.sqrt(3) + tol:
        return False
    return True

def _clip_to_triangle(p):
    """Clip a point to be inside the triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    # Clip y
    y = max(0, min(h, y))
    # Clip x based on y
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = max(x_min, min(x_max, x))
    return np.array([x, y])

def _nlp_feasibility(x0, alpha_target, max_iter=200, time_limit=5.0):
    """
    Try to find a configuration where all triangle areas >= alpha_target.
    Uses SLSQP with hard constraints on triangle areas and triangle boundaries.
    Returns (success, points) if found.
    """
    start_time = time.time()
    n = N
    
    # Precompute all triples
    triples = list(combinations(range(n), 3))
    n_triples = len(triples)
    
    def objective(x_flat):
        """Minimize sum of (alpha_target - area)^2 for violations, plus small regularization."""
        pts = x_flat.reshape(-1, 2)
        # Compute all triangle areas
        areas = np.zeros(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            areas[idx] = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        
        # Penalty for areas below target
        violations = np.maximum(0, alpha_target - areas)
        penalty = np.sum(violations ** 2)
        
        # Small regularization to avoid degenerate configurations
        reg = 1e-8 * np.sum(pts ** 2)
        
        return penalty + reg
    
    # Gradient of objective
    def objective_grad(x_flat):
        pts = x_flat.reshape(-1, 2)
        grad = np.zeros_like(x_flat)
        
        for idx, (i, j, k) in enumerate(triples):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            # Area = 0.5 * |cross(p2-p1, p3-p1)|
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            area = 0.5 * abs(cross)
            
            if area >= alpha_target:
                continue
            
            # Gradient of (alpha - area)^2 w.r.t. points
            # d/dp1: -2*(alpha-area) * d(area)/dp1
            # area = 0.5 * |cross|, so d(area)/dp = 0.5 * sign(cross) * d(cross)/dp
            sign = 1.0 if cross > 0 else -1.0
            
            # d(cross)/dp1 = (- (p3[1]-p1[1]), (p3[0]-p1[0]))
            # d(cross)/dp2 = ((p3[1]-p1[1]), -(p3[0]-p1[0]))
            # d(cross)/dp3 = (-(p2[1]-p1[1]), (p2[0]-p1[0]))
            
            coeff = -2.0 * (alpha_target - area) * 0.5 * sign
            
            d_p1 = np.array([-(p3[1] - p1[1]), (p3[0] - p1[0])])
            d_p2 = np.array([(p3[1] - p1[1]), -(p3[0] - p1[0])])
            d_p3 = np.array([-(p2[1] - p1[1]), (p2[0] - p1[0])])
            
            grad[2*i] += coeff * d_p1[0]
            grad[2*i+1] += coeff * d_p1[1]
            grad[2*j] += coeff * d_p2[0]
            grad[2*j+1] += coeff * d_p2[1]
            grad[2*k] += coeff * d_p3[0]
            grad[2*k+1] += coeff * d_p3[1]
        
        # Regularization gradient
        grad += 2e-8 * x_flat
        
        return grad
    
    # Constraints: points must be inside triangle
    # For each point, we need:
    # y >= 0
    # y <= h
    # x >= y/sqrt(3)  =>  sqrt(3)*x - y >= 0
    # x <= 1 - y/sqrt(3)  =>  sqrt(3)*x + y - sqrt(3) <= 0  =>  -(sqrt(3)*x + y - sqrt(3)) >= 0
    
    h = np.sqrt(3) / 2
    
    def make_constraints():
        cons = []
        for i in range(n):
            # y >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[2*i+1],
                'jac': lambda x, i=i: np.array([0, 1] if i==0 else [0]*2*i + [0,1] + [0]*(2*(n-i-1)))
            })
            # y <= h => h - y >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: h - x[2*i+1],
                'jac': lambda x, i=i: np.array([0, -1] if i==0 else [0]*2*i + [0,-1] + [0]*(2*(n-i-1)))
            })
            # sqrt(3)*x - y >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: np.sqrt(3)*x[2*i] - x[2*i+1],
                'jac': lambda x, i=i: np.array([np.sqrt(3), -1] if i==0 else [0]*2*i + [np.sqrt(3),-1] + [0]*(2*(n-i-1)))
            })
            # -(sqrt(3)*x + y - sqrt(3)) >= 0 => sqrt(3) - sqrt(3)*x - y >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: np.sqrt(3) - np.sqrt(3)*x[2*i] - x[2*i+1],
                'jac': lambda x, i=i: np.array([-np.sqrt(3), -1] if i==0 else [0]*2*i + [-np.sqrt(3),-1] + [0]*(2*(n-i-1)))
            })
        return cons
    
    constraints = make_constraints()
    
    try:
        result = minimize(
            objective, x0, jac=objective_grad,
            method='SLSQP', constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
        )
        
        # Check if we achieved the target
        pts = result.x.reshape(-1, 2)
        # Clip to triangle for safety
        pts = np.array([_clip_to_triangle(p) for p in pts])
        
        ma, _ = _min_triangle_area(pts)
        if ma >= alpha_target - 1e-8:
            return True, pts
        
        # Even if not fully feasible, return if improved
        return False, pts
        
    except Exception as e:
        return False, x0.reshape(-1, 2)

def _bisection_search(x0, alpha_lo, alpha_hi, time_limit=60.0):
    """
    Binary search for maximum feasible alpha.
    """
    start_time = time.time()
    best_points = x0.copy()
    best_alpha = alpha_lo
    
    # First, get a better initial estimate
    ma, _ = _min_triangle_area(x0)
    alpha_lo = max(alpha_lo, ma * 0.9)
    
    for iteration in range(15):  # ~15 bisection steps for good precision
        if time.time() - start_time > time_limit:
            break
        
        alpha_mid = (alpha_lo + alpha_hi) / 2
        
        # Try to find feasible configuration for alpha_mid
        success, pts = _nlp_feasibility(x0, alpha_mid, max_iter=150, time_limit=4.0)
        
        if time.time() - start_time > time_limit:
            break
        
        if success:
            # Feasible, try higher
            best_alpha = alpha_mid
            best_points = pts.copy()
            alpha_lo = alpha_mid
            x0 = pts.copy()  # Use this as new starting point
        else:
            # Not feasible, try lower
            alpha_hi = alpha_mid
    
    return best_points, best_alpha

def _generate_initial_configs(n_configs=5):
    """Generate diverse initial configurations."""
    configs = []
    h = np.sqrt(3) / 2
    
    # Config 1: Triangular lattice
    rows = [4, 3, 2, 2]
    pts = []
    total_rows = len(rows)
    for row_idx, count in enumerate(rows):
        y = h * (row_idx + 0.5) / total_rows
        x_start = y / np.sqrt(3)
        x_end = 1 - y / np.sqrt(3)
        for i in range(count):
            x = x_start + (x_end - x_start) * (i + 0.5) / count
            pts.append([x, y])
    configs.append(np.array(pts[:N]))
    
    # Config 2: Symmetric hexagonal
    pts2 = []
    pts2.append([0.5, h/3])  # center
    # 3 near corners
    eps = 0.08
    pts2.append([eps, eps*np.sqrt(3)])
    pts2.append([1-eps, eps*np.sqrt(3)])
    pts2.append([0.5, h - eps*np.sqrt(3)])
    # 6 around center
    for angle in range(6):
        theta = np.pi/3 * angle + np.pi/6
        r = 0.25
        x = 0.5 + r * np.cos(theta)
        y = h/3 + r * np.sin(theta)
        p = _clip_to_triangle([x, y])
        pts2.append(p)
    configs.append(np.array(pts2[:N]))
    
    # Config 3-5: Random with different seeds
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        pts3 = []
        attempts = 0
        while len(pts3) < N and attempts < 500:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = h * v
                p = _clip_to_triangle([x, y])
                # Check minimum distance to existing points
                if all(np.linalg.norm(p - q) > 0.08 for q in pts3):
                    pts3.append(p)
            attempts += 1
        if len(pts3) == N:
            configs.append(np.array(pts3))
    
    return configs

def construct():
    """Place 11 points to maximize minimum triangle area using NLP bisection."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_area = 0.0
    
    # Generate initial configurations
    configs = _generate_initial_configs(5)
    
    for config in configs:
        if time.time() > deadline - 20:
            break
        
        if len(config) != N:
            continue
        
        # Validate
        if not all(_inside_triangle(p, tol=1e-6) for p in config):
            continue
        
        ma, _ = _min_triangle_area(config)
        if ma > best_area:
            best_area = ma
            best_points = config.copy()
    
    # Use bisection + NLP to improve
    if best_points is not None and time.time() < deadline - 15:
        # Initial alpha estimate
        alpha_lo = best_area * 0.85
        alpha_hi = best_area * 1.3  # Upper bound
        
        # Run bisection from best starting point
        improved_pts, improved_alpha = _bisection_search(
            best_points, alpha_lo, alpha_hi, 
            time_limit=min(100, deadline - time.time() - 5)
        )
        
        if improved_alpha > best_area:
            best_area = improved_alpha
            best_points = improved_pts
    
    # Final validation and return
    if best_points is None:
        # Fallback
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95:
                points.append(_clip_to_triangle([u + 0.5*v, np.sqrt(3)/2 * v]))
        best_points = np.array(points)
    
    # Ensure all points are inside
    best_points = np.array([_clip_to_triangle(p) for p in best_points])
    
    return best_points
# EVOLVE-BLOCK-END
