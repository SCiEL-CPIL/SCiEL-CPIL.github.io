"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Global optimization via Simulated Annealing with a smooth penalty
for the minimum triangle area constraint. This avoids the local minima traps
of gradient-based methods on the non-smooth min() function by using a
differentiable approximation and stochastic global search.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _soft_min_objective(x_flat, temperature=1.0):
    """
    Smoothed objective: - (soft min of triangle areas) - penalty for being outside.
    Uses log-sum-exp approximation for the minimum to make it differentiable.
    """
    pts = x_flat.reshape(N, 2)
    
    # Penalty for being outside the triangle
    penalty = 0.0
    for p in pts:
        x, y = p
        # Distance to boundaries
        d1 = y # bottom
        d2 = np.sqrt(3) * x - y # left
        d3 = np.sqrt(3) * (1 - x) - y # right
        d4 = x # left vertical (not strictly needed if d2 handles it, but safe)
        d5 = 1 - x # right vertical
        
        # Sum of squared violations
        if d1 < 0: penalty += d1**2
        if d2 < 0: penalty += d2**2
        if d3 < 0: penalty += d3**2
        if d4 < 0: penalty += d4**2
        if d5 < 0: penalty += d5**2
        
    # Compute all triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
                
    areas = np.array(areas)
    
    # Soft minimum approximation: min(a) ~ (1/b) * log(sum(exp(-b*a)))
    # We want to maximize the min, so we minimize -min.
    # -min(a) ~ (1/b) * log(sum(exp(b*a))) is not quite right for max-min.
    # Let's use: f = - (1/b) * log( sum( exp(-b * area) ) )
    # As b -> inf, this approaches -min(area).
    b = 1.0 / temperature
    # Clip areas to avoid overflow in exp
    areas_clipped = np.clip(areas, 1e-6, None)
    exp_terms = np.exp(-b * areas_clipped)
    soft_min_val = (1.0 / b) * np.log(np.sum(exp_terms))
    
    # We want to maximize soft_min_val, so objective is -soft_min_val
    obj = -soft_min_val
    # Add penalty for being outside
    obj += 100.0 * penalty
    
    return obj

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _initial_guess(rng):
    """Generate an initial set of 11 points."""
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Simulated Annealing with a smooth objective function.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -float('inf')
    
    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 1e-5
    alpha = 0.995 # Cooling rate
    max_iter = 20000
    
    # Initial state
    current_pts = _initial_guess(rng)
    current_x = current_pts.flatten()
    current_obj = _soft_min_objective(current_x, temperature=T_start)
    
    # Track best
    if _check_inside(current_pts):
        area = _min_triangle_area(current_pts)
        if area > best_score:
            best_score = area
            best_pts = current_pts.copy()
    
    T = T_start
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Update temperature
        T *= alpha
        
        # Generate neighbor
        # Randomly perturb 1-2 points
        n_perturb = rng.choice([1, 1, 2])
        indices = rng.sample(range(N), n_perturb)
        
        new_x = current_x.copy()
        for idx in indices:
            # Gaussian perturbation
            delta = np.array([rng.gauss(0, T * 0.5), rng.gauss(0, T * 0.5)])
            new_x[idx*2 : idx*2+2] += delta
        
        new_pts = new_x.reshape(N, 2)
        
        # Project to triangle to stay feasible or near-feasible
        new_pts_proj = _project_to_triangle(new_pts)
        new_x_proj = new_pts_proj.flatten()
        
        # Evaluate objective
        new_obj = _soft_min_objective(new_x_proj, temperature=T)
        
        # Acceptance criterion
        delta_obj = new_obj - current_obj
        if delta_obj < 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-10)):
            current_x = new_x_proj
            current_pts = new_pts_proj
            current_obj = new_obj
            
            # Update best if valid and better
            if _check_inside(current_pts):
                area = _min_triangle_area(current_pts)
                if area > best_score:
                    best_score = area
                    best_pts = current_pts.copy()
        
        # Periodic local polish to escape local minima
        if it % 1000 == 0 and it > 0:
            # Try a few random restarts or larger jumps
            for _ in range(5):
                jump_pts = _initial_guess(rng)
                jump_x = jump_pts.flatten()
                jump_obj = _soft_min_objective(jump_x, temperature=T)
                # Accept if better or with probability
                d_obj = jump_obj - current_obj
                if d_obj < 0 or rng.random() < np.exp(-d_obj / max(T, 1e-10)):
                    current_x = jump_x
                    current_pts = jump_pts
                    current_obj = jump_obj
                    if _check_inside(current_pts):
                        area = _min_triangle_area(current_pts)
                        if area > best_score:
                            best_score = area
                            best_pts = current_pts.copy()
                
                if time.time() - start_time > DEADLINE - 5:
                    break

    # Final polish: try to improve the best found point with a local optimizer
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        # Use L-BFGS-B on the best point with the soft objective for a final push
        try:
            result = minimize(
                lambda x: _soft_min_objective(x, temperature=0.01),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100}
            )
            polished_pts = result.x.reshape(N, 2)
            polished_pts = _project_to_triangle(polished_pts)
            if _check_inside(polished_pts):
                area = _min_triangle_area(polished_pts)
                if area > best_score:
                    best_score = area
                    best_pts = polished_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = _initial_guess(rng)
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts
