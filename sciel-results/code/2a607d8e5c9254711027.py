"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas_vec(pts):
    """
    Vectorized computation of areas of all triangles formed by triples of points.
    pts: (N, 2)
    Returns: array of shape (N*(N-1)*(N-2)/6,)
    """
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas[idx] = area
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the point set."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas_vec(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min(areas, temperature):
    """
    Compute the soft minimum of an array.
    soft_min(a, T) = -1/T * log(sum(exp(-T * a)))
    """
    min_a = np.min(areas)
    shifted = areas - min_a
    # Numerical stability: exp(-T * shifted) where shifted >= 0
    log_sum = np.log(np.sum(np.exp(-temperature * shifted)))
    return min_a - (1.0 / temperature) * log_sum

def _objective_smooth(pts_flat, temperature):
    """
    Objective function to minimize: -soft_min(areas) / hull_area
    """
    pts = pts_flat.reshape(-1, 2)
    areas = _triangle_areas_vec(pts)
    
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10 # Large penalty
    
    sm = _soft_min(areas, temperature)
    # We want to maximize sm / hull_area
    # So we minimize -sm / hull_area
    return -sm / hull_area

def _polish_l_bfgs(pts, time_limit, T=100.0):
    """
    Polishing step using L-BFGS-B on the smoothed objective.
    Returns improved points and score.
    """
    x0 = pts.flatten()
    
    def obj(x):
        return _objective_smooth(x, T)
        
    try:
        # Use a short time limit for polishing
        result = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        new_pts = result.x.reshape(-1, 2)
        new_score = _score(new_pts)
        return new_pts, new_score
    except Exception:
        return pts, _score(pts)

def _simulated_annealing_step(current_pts, current_score, temperature, rng, step_size):
    """
    Perform a single simulated annealing step.
    """
    # Generate a perturbation
    # Step size should be adaptive or fixed relative to the domain
    noise = rng.normal(0, step_size, current_pts.shape)
    new_pts = current_pts + noise
    
    # Optional: keep points within a reasonable bound (e.g., unit square or circle)
    # The problem is scale invariant, but bounding helps numerical stability
    # Let's clip to [-2, 2] to prevent drift
    new_pts = np.clip(new_pts, -2.0, 2.0)
    
    new_score = _score(new_pts)
    
    # Accept if better, or with probability exp((new_score - current_score) / temperature)
    if new_score > current_score:
        return new_pts, new_score, True
    else:
        # Avoid log(0) or negative probability
        delta = new_score - current_score
        if delta == 0:
            prob = 1.0
        else:
            prob = np.exp(delta / temperature)
        if rng.random() < prob:
            return new_pts, new_score, True
        else:
            return current_pts, current_score, False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for Python overhead
    
    best_pts = None
    best_score = -1.0
    
    rng = np.random.default_rng(42)
    
    # Initializations
    # 1. Regular Polygon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # 2. Random points in a square
    init_rand = rng.uniform(-1, 1, (N, 2))
    
    initializations = [init_reg, init_rand]
    
    # Add a few more random convex-ish inits
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)

    # Simulated Annealing Parameters
    # We will run SA for a few seconds, then polish, then restart.
    # The "score" is the min area / hull area. It is scale invariant.
    # We need to tune the step_size and temperature.
    
    num_restarts = 10
    sa_time_per_restart = 8.0  # seconds
    polish_time_per_restart = 2.0
    
    for restart_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - (sa_time_per_restart + polish_time_per_restart):
            break
            
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
        # Simulated Annealing Loop
        # Start with high temperature, cool down
        # The "energy" is -score. We want to minimize energy.
        # Since score is positive, we maximize score.
        # SA acceptance: exp((new_score - old_score) / T)
        # If new_score < old_score, exponent is negative.
        
        start_temp = 0.05  # Initial temperature
        end_temp = 0.001   # Final temperature
        # We need to estimate the scale of score changes.
        # Typical score is ~0.01-0.05. Changes are small.
        
        sa_start = time.time()
        sa_deadline = sa_start + sa_time_per_restart
        
        # Adaptive step size: start small, maybe increase if stuck?
        # For now, fixed step size relative to the "size" of the configuration.
        # The configuration is roughly unit scale.
        step_size = 0.05
        
        # We will do a geometric cooling schedule
        # T(t) = T_start * (T_end/T_start)^(t/T_total)
        # But we don't know T_total exactly, so we just decrement T each step or use a simple schedule.
        # Let's use a simple linear or exponential decay based on time elapsed.
        
        # Instead of fixed steps, we run until time runs out.
        # We need to track temperature. Let's just decay it based on iteration count or time.
        # A simple approach: T = T_start * (1 - elapsed/sa_time)
        
        while time.time() < sa_deadline:
            elapsed = time.time() - sa_start
            # Cooling schedule: linear from start_temp to end_temp
            progress = min(1.0, elapsed / sa_time_per_restart)
            current_temp = start_temp * (1.0 - progress) + end_temp * progress
            
            # Step size can also be annealed, but let's keep it constant for simplicity
            # or slightly reduce it
            current_step = step_size * (1.0 - 0.5 * progress)
            
            current_pts, current_score, accepted = _simulated_annealing_step(
                current_pts, current_score, current_temp, rng, current_step
            )
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
            # Occasionally polish to snap to local optimum?
            # This might be expensive. Let's do it only at the end of SA phase.
            
        # Polish the result of SA
        if time.time() < deadline - polish_time_per_restart:
            polished_pts, polished_score = _polish_l_bfgs(current_pts, polish_time_per_restart)
            if polished_score > current_score:
                current_pts = polished_pts
                current_score = polished_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
    # Final Polish on the global best
    if best_pts is not None and time.time() < deadline - 5.0:
        # Use a higher temperature for the soft min to allow larger moves?
        # Or just standard polish.
        # Let's try a few different temperatures for the soft min polish
        for T_polish in [50.0, 200.0, 500.0]:
            if time.time() > deadline - 2.0:
                break
            polished_pts, polished_score = _polish_l_bfgs(best_pts, 2.0, T=T_polish)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
                
    # Fallback
    if best_pts is None:
        best_pts = initializations[0]
        
    return best_pts

# EVOLVE-BLOCK-END
