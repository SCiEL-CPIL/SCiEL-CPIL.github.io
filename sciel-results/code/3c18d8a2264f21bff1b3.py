import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        # Check for collinearity or degenerate hull
        if np.linalg.matrix_rank(points - points[0]) < 2:
            return 0.0

        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _params_to_points(params):
    """
    Convert 6 parameters [a0, a1, a2, b0, b1, b2] to N points on a quadratic curve.
    x(t) = a0 + a1*t + a2*t^2
    y(t) = b0 + b1*t + b2*t^2
    t is uniformly spaced in [0, 1].
    """
    a0, a1, a2, b0, b1, b2 = params
    t = np.linspace(0, 1, N)
    xs = a0 + a1 * t + a2 * t**2
    ys = b0 + b1 * t + b2 * t**2
    return np.column_stack([xs, ys])

def _objective(params):
    """
    Objective function for minimization (returns negative score).
    """
    points = _params_to_points(params)
    
    # Additional check: ensure points are distinct enough
    # If points are too close, score will be low/zero, which is fine for minimization
    score = _score(points)
    
    # If score is 0 due to exception or degeneracy, return a large penalty to push away
    # However, _score returns 0.0. Minimizing -0.0 is 0.0.
    # We want to maximize score, so minimize -score.
    # If score is 0, -score is 0. This might be a local minimum trap if we start bad.
    # To help, we can add a small penalty for degeneracy if score is 0.
    if score == 0.0:
        # Check if it's actually degenerate
        if np.linalg.matrix_rank(points - points[0]) < 2:
            return 1.0 # Large positive value to minimize
        else:
            return 0.0
            
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Strategy: Optimize the 6 coefficients of the quadratic curve.
    # We try multiple initializations to escape local minima.
    
    # Initialization 1: Standard Parabola y = x^2, x in [-1, 1]
    # x = -1 + 2t, y = (-1+2t)^2 = 1 - 4t + 4t^2
    # a0=-1, a1=2, a2=0
    # b0=1, b1=-4, b2=4
    init1 = np.array([-1.0, 2.0, 0.0, 1.0, -4.0, 4.0])
    
    # Initialization 2: Flatter parabola
    # x = -1 + 2t, y = 0.5 * x^2
    init2 = np.array([-1.0, 2.0, 0.0, 0.5, -2.0, 2.0])
    
    # Initialization 3: Slightly rotated/skewed parabola
    # x = -1 + 2t + 0.5t^2, y = 1 - 4t + 4t^2
    init3 = np.array([-1.0, 2.0, 0.5, 1.0, -4.0, 4.0])
    
    # Initialization 4: Regular polygon approximation? 
    # A regular polygon is not a quadratic curve. 
    # But we can try a "U-shape" that is close to a circle segment.
    # Circle: x^2 + y^2 = R^2. y = sqrt(R^2 - x^2) ~ R - x^2/2R.
    # Let R=1, x in [-1,1]. y ~ 1 - 0.5 x^2.
    # x = -1 + 2t.
    # y ~ 1 - 0.5 (-1+2t)^2 = 1 - 0.5 (1 - 4t + 4t^2) = 0.5 + 2t - 2t^2.
    init4 = np.array([-1.0, 2.0, 0.0, 0.5, 2.0, -2.0])
    
    initializations = [init1, init2, init3, init4]
    
    # We will run Powell's method for each initialization with a time budget.
    # Powell is robust for low-dimensional problems.
    
    time_per_opt = 25.0 # 4 opts * 25s = 100s. Leaves buffer.
    
    for i, x0 in enumerate(initializations):
        current_time = time.time()
        if current_time > deadline - 5:
            break
            
        remaining = deadline - current_time - 2
        if remaining < 5:
            continue
            
        # Cap the time for this specific optimization
        opt_time_budget = min(time_per_opt, remaining)
        
        try:
            # Powell's method doesn't have a direct 'max time' option in scipy.minimize
            # We rely on maxiter and maxfev, or just let it run and check time later.
            # However, Powell can be slow if it wanders. 
            # Using 'Nelder-Mead' might be faster to converge in 6D for smooth-ish landscapes.
            # Let's try Nelder-Mead first as it was used in the parent and is often faster for <10D.
            
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            # Check if we timed out (approximate)
            if time.time() > deadline - 5:
                # Still evaluate the result found so far
                pass
                
            best_params = res.x
            points = _params_to_points(best_params)
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception as e:
            # If optimization fails, ignore and try next
            continue
            
    # Fallback: Regular 14-gon if optimization failed or score is very low
    if best_points is None or best_score < 1e-6:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
        
    # One final polish: Try to refine the best points found using full 28D optimization?
    # The prompt asks for a DIFFERENT strategy. The quadratic curve is the strategy.
    # Refining with 28D might be seen as "refining the current program" if the current program
    # did that, but the current program did ring optimization. 
    # However, 28D optimization is very expensive and risky in 120s.
    # The quadratic curve constraint reduces dimensionality from 28 to 6, which is the key advantage.
    
    return best_points

# EVOLVE-BLOCK-END
