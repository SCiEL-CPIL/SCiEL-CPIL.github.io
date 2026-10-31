import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    # y >= 0
    v1 = np.maximum(0, -y)
    # y <= sqrt(3) * x
    v2 = np.maximum(0, y - S3 * x)
    # y <= sqrt(3) * (1 - x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _clamp_to_triangle(points):
    """
    Project points to the nearest point inside the triangle.
    This is an approximation but helps keep the search valid.
    """
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # y >= 0
    y = np.maximum(y, 0)
    
    # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    
    # y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
        
    # Re-check y >= 0 after x adjustments (shouldn't change but safe)
    y = np.maximum(y, 0)
    
    # Sometimes projection can go out of bounds, so do a final safety clip to box [0,1]x[0,H]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    
    # Final check and correction
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
        
    return np.column_stack((x, y))

def _smooth_objective(flat_pts, beta=100.0):
    """
    Smooth approximation of -min_area.
    Used for Differential Evolution.
    """
    points = flat_pts.reshape(-1, 2)
    points = _clamp_to_triangle(points)
    
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp approximation of min
    # min(A) approx -1/beta * log(sum(exp(-beta * A)))
    # We want to minimize -min(A), so we return 1/beta * log(sum(exp(-beta * A)))
    # But we also need to handle the penalty.
    # If any point is outside, penalty is large.
    
    pen = _penalty(points)
    
    if np.any(areas < 1e-10):
         # Avoid log(0) issues, though areas shouldn't be 0 if points are distinct
         # But if points coincide, area is 0.
         # exp(-beta * 0) = 1.
         pass
    
    # Shift areas to avoid underflow/overflow in exp
    # Actually, for small areas, exp(-beta * A) is close to 1.
    # For large areas, it's close to 0.
    # The sum is dominated by the smallest area.
    
    # Numerical stability:
    # Let m = min(areas).
    # sum(exp(-beta * A)) = exp(-beta * m) * sum(exp(-beta * (A - m)))
    # log(sum) = -beta * m + log(sum(exp(-beta * (A - m))))
    # -1/beta * log(sum) = m - 1/beta * log(sum(exp(-beta * (A - m))))
    
    m = np.min(areas)
    shifted = areas - m
    # exp(-beta * shifted) where shifted >= 0
    # If beta is large, exp(-beta * shifted) is 0 for shifted > 0.
    # So sum is approx 1. log(1) = 0.
    # Result approx m.
    
    # To be safe and smooth:
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = m - log_sum_exp / beta
    
    # We want to MINIMIZE -smooth_min.
    # So objective is -smooth_min + penalty_term.
    # Penalty term: if points are outside, we add a large value.
    # The _penalty function returns squared distance.
    # We want to push points inside.
    
    obj = -smooth_min + 100.0 * pen
    
    return obj

def _hard_objective(flat_pts):
    """
    Hard objective: -min_area + penalty.
    Used for Nelder-Mead.
    """
    points = flat_pts.reshape(-1, 2)
    points = _clamp_to_triangle(points)
    
    area = _min_area(points)
    pen = _penalty(points)
    
    return -area + 100.0 * pen

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_population_structured():
    """
    Generate a structured initial configuration.
    """
    points = []
    # Vertices
    points.append(np.array([0.0, 0.0]))
    points.append(np.array([1.0, 0.0]))
    points.append(np.array([0.5, H]))
    # Midpoints
    points.append(np.array([0.5, 0.0]))
    points.append(np.array([0.25, H/2]))
    points.append(np.array([0.75, H/2]))
    # Center
    points.append(np.array([0.5, H/3]))
    # Inner ring
    remaining = 11 - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        r = 0.1
        cx = 0.5 + r * np.cos(angle)
        cy = H/3 + r * np.sin(angle) * (H/0.5)
        points.append(np.array([cx, cy]))
    
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 150.0 # Evaluator limit
    safety_margin = 10.0
    effective_timeout = timeout - safety_margin
    
    rng = np.random.default_rng(42)
    
    # Bounds for DE
    # x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution
    # DE is slow, so we limit maxiter and popsize.
    # We use a smooth objective to guide it.
    de_time_limit = effective_timeout * 0.7
    
    print(f"Starting DE with time limit {de_time_limit}s")
    
    try:
        # We need to stop DE before it finishes if it takes too long.
        # scipy DE doesn't have a direct time limit, but we can monitor via callback.
        # However, callback is called per iteration.
        # Let's just set reasonable maxiter.
        # 11 points -> 22 variables.
        # popsize=15 (default) -> 15 * 22 = 330 evaluations per generation.
        # If we allow 50 generations, that's 16500 evaluations.
        # Each evaluation is fast (vectorized).
        
        de_result = differential_evolution(
            _smooth_objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=rng,
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        if de_result.success or de_result.fun < np.inf:
            de_pts = _clamp_to_triangle(de_result.x.reshape(-1, 2))
            de_score = _min_area(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_points = de_pts
                print(f"DE Best Score: {de_score}")
                
    except Exception as e:
        print(f"DE Error: {e}")
    
    # If DE failed or produced nothing, use structured init
    if best_points is None:
        best_points = _initial_population_structured()
        best_score = _min_area(best_points)
        print(f"Using Structured Init Score: {best_score}")

    # Phase 2: Nelder-Mead Polish
    # We will run NM on the hard objective.
    # NM is derivative-free and good for non-smooth functions.
    
    remaining_time = effective_timeout - (time.time() - start_time)
    
    if remaining_time > 5:
        print(f"Starting Nelder-Mead polish with time limit {remaining_time}s")
        
        # We can try multiple restarts from slightly perturbed best_points
        # Or just one long run.
        # Let's do a single run with a generous maxiter.
        
        def nm_obj(flat_pts):
            return _hard_objective(flat_pts)
        
        try:
            nm_result = minimize(
                nm_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            nm_pts = _clamp_to_triangle(nm_result.x.reshape(-1, 2))
            nm_score = _min_area(nm_pts)
            
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_pts
                print(f"NM Best Score: {best_score}")
                
        except Exception as e:
            print(f"NM Error: {e}")
            
        # If time remains, try a second NM run from a slightly different start
        # e.g., the second best from DE? We don't have that easily.
        # Let's just do another NM run if we improved significantly or if time is plenty.
        
        remaining_time2 = effective_timeout - (time.time() - start_time)
        if remaining_time2 > 10:
             # Perturb the best points slightly and run NM again
             perturbed = best_points + rng.normal(0, 0.01, size=best_points.shape)
             perturbed = _clamp_to_triangle(perturbed)
             
             try:
                nm_result2 = minimize(
                    nm_obj,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                nm_pts2 = _clamp_to_triangle(nm_result2.x.reshape(-1, 2))
                nm_score2 = _min_area(nm_pts2)
                
                if nm_score2 > best_score:
                    best_score = nm_score2
                    best_points = nm_pts2
                    print(f"NM2 Best Score: {best_score}")
             except Exception as e:
                print(f"NM2 Error: {e}")

    # Final validity check
    if best_points is None:
        best_points = _initial_population_structured()
        
    best_points = _clamp_to_triangle(best_points)
    
    # Final score check
    final_score = _min_area(best_points)
    print(f"Final Score: {final_score}")
    
    return best_points

if __name__ == "__main__":
    construct()
