"""Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Simulated Annealing with Soft-Min objective.
"""
import numpy as np
import time
import random
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # seconds, safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points.
    Returns the raw area (not normalized)."""
    n = len(pts)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        area = 0.5 * cross
        if area < min_area:
            min_area = area
    return min_area

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array."""
    n = len(pts)
    areas = []
    for i, j, k in combinations(range(n), 3):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        areas.append(0.5 * cross)
    return np.array(areas)

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

def _soft_min(areas, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    areas_clipped = np.clip(areas, 1e-12, None)
    neg_beta_a = -beta * areas_clipped
    max_val = np.max(neg_beta_a)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_a - max_val)))
    return -log_sum / beta

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

def _structured_guess(rng):
    """Generate a structured initial guess using vertices, edge midpoints, and center."""
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, S3H])
    
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2
    
    centroid = (v0 + v1 + v2) / 3
    
    pts = [v0, v1, v2, m01, m12, m20, centroid]
    for _ in range(4):
        pts.append(_random_point_in_triangle(rng))
    
    return np.array(pts)

def _initial_guess(rng, structured=False):
    """Generate an initial set of 11 points."""
    if structured:
        return _structured_guess(rng)
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Initial population
    # We will run multiple independent SA chains or one long chain with restarts.
    # Given the time limit, a single robust SA with periodic restarts if stuck is good.
    # Or, run several parallel chains in sequence.
    
    num_chains = 10
    steps_per_chain = 5000
    
    # Adaptive beta: start low (smooth), end high (sharp)
    beta_start = 5.0
    beta_end = 100.0
    
    # Temperature schedule for SA
    # We are maximizing, so we want to accept worse solutions (lower min area) with probability exp((new-old)/T)
    # Let's use a log-temperature decay.
    T_start = 0.05
    T_end = 0.001
    
    for chain in range(num_chains):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Initialize chain
        if chain == 0:
            current_pts = _initial_guess(rng, structured=True)
        else:
            # Restart from best so far with some perturbation, or random
            if rng.random() < 0.5 and best_pts is not None:
                current_pts = best_pts.copy()
                # Perturb
                for i in range(N):
                    current_pts[i, 0] += rng.gauss(0, 0.05)
                    current_pts[i, 1] += rng.gauss(0, 0.05)
                current_pts = _project_to_triangle(current_pts)
            else:
                current_pts = _initial_guess(rng, structured=(chain % 3 == 0))
        
        current_area = _min_triangle_area(current_pts)
        if current_area > best_area:
            best_area = current_area
            best_pts = current_pts.copy()
        
        # SA Loop
        for step in range(steps_per_chain):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Schedule parameters
            progress = step / steps_per_chain
            # Beta increases to sharpen the soft-min
            current_beta = beta_start * (beta_end / beta_start) ** progress
            # Temperature decreases
            current_T = T_start * (T_end / T_start) ** progress
            
            # Generate neighbor
            # Move a random point by a random small step
            idx = rng.randint(0, N - 1)
            step_size = 0.05 * (1.0 - progress * 0.5) # Reduce step size slightly
            dx = rng.gauss(0, step_size)
            dy = rng.gauss(0, step_size)
            
            new_pts = current_pts.copy()
            new_pts[idx, 0] += dx
            new_pts[idx, 1] += dy
            new_pts = _project_to_triangle(new_pts)
            
            if not _check_inside(new_pts):
                continue
            
            # Compute new objective
            # For speed, we can use soft-min for the decision, but track exact min for best.
            # However, soft-min is expensive too. 
            # Let's use exact min for the decision as well, since N=11 is small (165 triangles).
            new_area = _min_triangle_area(new_pts)
            
            # Acceptance criterion
            if new_area >= current_area:
                current_pts = new_pts
                current_area = new_area
                if current_area > best_area:
                    best_area = current_area
                    best_pts = current_pts.copy()
            else:
                # Metropolis acceptance
                delta = current_area - new_area # positive value
                # Probability = exp(-delta / T)
                # If delta/T is large, prob is small.
                try:
                    prob = np.exp(-delta / current_T)
                except OverflowError:
                    prob = 0.0
                
                if rng.random() < prob:
                    current_pts = new_pts
                    current_area = new_area
                    if current_area > best_area:
                        best_area = current_area
                        best_pts = current_pts.copy()
        
        # After each chain, do a quick local polish using L-BFGS on the soft-min
        # to squeeze out the last bit of precision.
        if best_pts is not None and time.time() - start_time < DEADLINE - 15:
            try:
                from scipy.optimize import minimize
                
                def _smooth_objective(x_flat, beta=100.0):
                    pts = x_flat.reshape(N, 2)
                    if not _check_inside(pts, eps=1e-6):
                        return 1e10
                    areas = _all_triangle_areas(pts)
                    return -_soft_min(areas, beta)
                
                x0 = best_pts.flatten()
                result = minimize(
                    lambda x: _smooth_objective(x, beta=100.0),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12}
                )
                opt_pts = result.x.reshape(N, 2)
                opt_pts = _project_to_triangle(opt_pts)
                if _check_inside(opt_pts):
                    area = _min_triangle_area(opt_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = opt_pts.copy()
            except Exception:
                pass

    if best_pts is None:
        pts = _initial_guess(rng)
        return pts
    
    return best_pts
