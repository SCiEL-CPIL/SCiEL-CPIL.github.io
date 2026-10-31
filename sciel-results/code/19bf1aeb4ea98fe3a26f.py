"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    if y < -tol:
        return False
    if y > 2*h*x + tol:
        return False
    if y > 2*h*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p, tol) for p in points)

def _negative_min_area(params):
    """Objective for differential evolution: negative of min triangle area."""
    points = params.reshape(-1, 2)
    if not _all_inside(points):
        penalty = 0.0
        for p in points:
            h = np.sqrt(3) / 2
            x, y = p[0], p[1]
            if x < 0 or x > 1 or y < 0 or y > h:
                penalty += 10.0
            if y > 2*h*x + 1e-4:
                penalty += 10.0
            if y > 2*h*(1-x) + 1e-4:
                penalty += 10.0
        return -0.001 + penalty
    return -_min_triangle_area(points)

def _local_polish(points, deadline):
    """Local refinement using Nelder-Mead and coordinate descent."""
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    for pass_num in range(3):
        if time.time() > deadline - 5:
            break
        
        def objective(params):
            pts = params.reshape(-1, 2)
            if not _all_inside(pts):
                return 10.0
            return -_min_triangle_area(pts)
        
        try:
            result = minimize(objective, points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success or result.fun < -current_min_area:
                new_points = result.x.reshape(-1, 2)
                if _all_inside(new_points):
                    new_min_area = _min_triangle_area(new_points)
                    if new_min_area > current_min_area:
                        points = new_points
                        current_min_area = new_min_area
        except Exception:
            pass
        
        if time.time() > deadline - 3:
            break
        
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                for coord in range(2):
                    for step in [0.01, 0.005, 0.001, 0.0005]:
                        old_val = points[i, coord]
                        points[i, coord] = old_val + step
                        if _all_inside(points):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val - step
                        if _all_inside(points):
                            new_area = _min_triangle_area(points)
                            if new_area > current_min_area:
                                current_min_area = new_area
                                improved = True
                                continue
                        points[i, coord] = old_val
            if not improved:
                break
    
    return points, current_min_area

def _nlp_solve(points_init, deadline):
    """
    NLP feasibility approach: maximize alpha subject to all triangle areas >= alpha.
    Uses SLSQP with constraints.
    """
    if time.time() > deadline - 10:
        return points_init, _min_triangle_area(points_init)
    
    def objective(params):
        points = params.reshape(-1, 2)
        if not _all_inside(points, tol=1e-4):
            return 100.0
        return -_min_triangle_area(points)
    
    h = np.sqrt(3) / 2
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, h))
    
    def _constraint_left(params):
        points = params.reshape(-1, 2)
        return np.array([2*h*points[i,0] - points[i,1] for i in range(N)])
    
    def _constraint_right(params):
        points = params.reshape(-1, 2)
        return np.array([2*h*(1-points[i,0]) - points[i,1] for i in range(N)])
    
    constraints = [
        {'type': 'ineq', 'fun': _constraint_left},
        {'type': 'ineq', 'fun': _constraint_right}
    ]
    
    try:
        result = minimize(objective, points_init.flatten(), method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-10})
        
        if result.fun < 100.0:
            new_points = result.x.reshape(-1, 2)
            if _all_inside(new_points, tol=1e-3):
                new_min_area = _min_triangle_area(new_points)
                return new_points, new_min_area
    except Exception:
        pass
    
    return points_init, _min_triangle_area(points_init)

def _build_structured_config():
    """Build a structured 11-point configuration based on known Heilbronn-type patterns."""
    h = np.sqrt(3) / 2
    points = []
    
    eps = 0.02
    points.append([eps, eps])
    points.append([1-eps, eps])
    points.append([0.5, h - eps*2])
    
    points.append([0.5, eps])
    points.append([1-eps, h/2])
    points.append([eps, h/2])
    
    points.append([0.5, h/3])
    
    cx, cy = 0.5, h/3
    r = 0.15
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi])
    for a in angles:
        px = cx + r*np.cos(a)
        py = cy + r*np.sin(a)*0.9
        if _inside_triangle([px, py], tol=1e-3):
            points.append([px, py])
    
    return np.array(points[:N])

def _simulated_annealing(points_init, deadline, seed=42):
    """
    Simulated annealing on point positions to escape local minima.
    Uses a smooth approximation of the min-area objective.
    """
    rng = np.random.RandomState(seed)
    points = points_init.copy()
    current_area = _min_triangle_area(points)
    best_points = points.copy()
    best_area = current_area
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Pick a random point and perturb it
        i = rng.randint(N)
        dx = rng.normal(0, 0.05)
        dy = rng.normal(0, 0.05)
        
        old_x, old_y = points[i, 0], points[i, 1]
        new_x = np.clip(old_x + dx, 0, 1)
        new_y = np.clip(old_y + dy, 0, np.sqrt(3)/2)
        
        # Check if new position is inside triangle
        if not _inside_triangle([new_x, new_y], tol=1e-4):
            continue
        
        points[i, 0] = new_x
        points[i, 1] = new_y
        
        new_area = _min_triangle_area(points)
        delta = new_area - current_area
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_area = new_area
            if new_area > best_area:
                best_area = new_area
                best_points = points.copy()
        else:
            points[i, 0] = old_x
            points[i, 1] = old_y
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_points, best_area

def _log_sum_exp_smooth(points, temperature=0.01):
    """
    Smooth approximation of min area using log-sum-exp.
    This provides a differentiable objective for gradient-based methods.
    """
    n = len(points)
    areas = np.zeros(165)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas[idx] = area
                idx += 1
    
    # Log-sum-exp for minimum: -log-sum-exp(-areas/T) * T
    max_neg = np.max(-areas / temperature)
    log_sum = max_neg + np.log(np.sum(np.exp(-areas / temperature - max_neg)))
    return -log_sum * temperature

def _gradient_polish(points_init, deadline):
    """
    Gradient-based polishing using L-BFGS-B with smooth log-sum-exp objective.
    This can escape local minima better than Nelder-Mead.
    """
    if time.time() > deadline - 5:
        return points_init, _min_triangle_area(points_init)
    
    h = np.sqrt(3) / 2
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, h))
    
    def smooth_objective(params):
        pts = params.reshape(-1, 2)
        # Check bounds
        for i in range(N):
            x, y = pts[i, 0], pts[i, 1]
            if y > 2*h*x + 1e-3 or y > 2*h*(1-x) + 1e-3:
                return 10.0
        return -_log_sum_exp_smooth(pts, temperature=0.005)
    
    def constraint_left(params):
        pts = params.reshape(-1, 2)
        return np.array([2*h*pts[i,0] - pts[i,1] for i in range(N)])
    
    def constraint_right(params):
        pts = params.reshape(-1, 2)
        return np.array([2*h*(1-pts[i,0]) - pts[i,1] for i in range(N)])
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_left},
        {'type': 'ineq', 'fun': constraint_right}
    ]
    
    try:
        result = minimize(smooth_objective, points_init.flatten(), method='L-BFGS-B',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < 10.0:
            new_points = result.x.reshape(-1, 2)
            if _all_inside(new_points, tol=1e-3):
                new_min_area = _min_triangle_area(new_points)
                return new_points, new_min_area
    except Exception:
        pass
    
    return points_init, _min_triangle_area(points_init)

def construct():
    deadline = time.time() + 135
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Differential evolution for global search
    if time.time() < deadline - 40:
        bounds = [(0, 1)] * 2 * N
        try:
            de_result = differential_evolution(
                _negative_min_area, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='sobol'
            )
            de_points = de_result.x.reshape(-1, 2)
            if _all_inside(de_points):
                de_min_area = _min_triangle_area(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 2: Structured configuration + NLP
    if time.time() < deadline - 30:
        struct_pts = _build_structured_config()
        nlp_pts, nlp_area = _nlp_solve(struct_pts, deadline)
        if nlp_area > best_min_area:
            best_min_area = nlp_area
            best_points = nlp_pts.copy()
    
    # Strategy 3: Simulated annealing from multiple starts
    if time.time() < deadline - 25:
        starts = []
        
        # Start 1: structured
        if best_points is not None:
            starts.append(best_points.copy())
        starts.append(_build_structured_config())
        
        # Start 2: random
        rng = np.random.RandomState(123)
        rand_pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.95 and u > 0.02 and v > 0.02:
                    px = u + 0.5 * v
                    py = h * v
                    rand_pts.append([px, py])
                    break
        starts.append(np.array(rand_pts))
        
        for s in starts:
            if time.time() > deadline - 15:
                break
            sa_pts, sa_area = _simulated_annealing(s, deadline, seed=42)
            if sa_area > best_min_area:
                best_min_area = sa_area
                best_points = sa_pts.copy()
    
    # Strategy 4: Gradient polishing with log-sum-exp smooth
    if best_points is not None and time.time() < deadline - 10:
        grad_pts, grad_area = _gradient_polish(best_points, deadline)
        if grad_area > best_min_area:
            best_min_area = grad_area
            best_points = grad_pts.copy()
    
    # Final local polish
    if best_points is not None and time.time() < deadline - 3:
        final_pts, final_area = _local_polish(best_points, deadline)
        if final_area > best_min_area:
            best_min_area = final_area
            best_points = final_pts.copy()
    
    if best_points is None:
        best_points = _build_structured_config()
    
    return best_points
# EVOLVE-BLOCK-END
