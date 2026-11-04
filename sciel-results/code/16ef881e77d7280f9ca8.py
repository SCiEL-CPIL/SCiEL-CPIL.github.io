"""
Optimized solver for the Heilbronn problem with n=14.
Strategy:
1. Two-ring initialization (7 outer, 7 inner).
2. Soft-min (log-sum-exp) optimization with L-BFGS-B for smooth gradient descent.
3. Exact min-area polish with Nelder-Mead for final refinement.
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized computation of all triangle areas
    # n=14, so C(14,3) = 364 triangles. This is small enough to do in one shot.
    i, j, k = np.indices((N, N, N))
    # Filter valid triples i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    crosses = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(crosses)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_score(pts, beta=50.0):
    """
    Approximate the min triangle area using log-sum-exp for differentiability.
    Score = -soft_min(areas) / hull_area
    We want to maximize the score, so we minimize -score.
    """
    if len(pts) < 3:
        return 1e9
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e9
        
    if hull_area < 1e-12:
        return 1e9
        
    i, j, k = np.indices((N, N, N))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    crosses = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(crosses)
    
    # Log-sum-exp trick for stability
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize min(areas), so we minimize -soft_min(areas).
    # Let f = -soft_min(areas) / hull_area.
    # -soft_min(areas) = 1/beta * log(sum(exp(-beta * areas)))
    
    scaled_areas = -beta * areas
    max_scaled = np.max(scaled_areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled_areas - max_scaled)))
    soft_min_val = -log_sum / beta
    
    # We want to maximize soft_min_val / hull_area
    # So we minimize -soft_min_val / hull_area
    return -soft_min_val / hull_area

def _neg_score(pts_flat):
    """
    Objective function for minimization.
    Reshapes flat array to (14, 2) and returns negative exact score.
    """
    pts = pts_flat.reshape(N, 2)
    s = _score(pts)
    if s == 0.0:
        return 1e9
    return -s

def _neg_soft_score(pts_flat, beta=50.0):
    """
    Objective function for soft-min minimization.
    """
    pts = pts_flat.reshape(N, 2)
    return _soft_min_score(pts, beta)

def _generate_two_ring(r, theta_offset):
    """
    Generate initial points: 7 outer on unit circle, 7 inner on circle of radius r.
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta_offset
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -float('inf')
    best_pts = None
    
    # Initial guesses for (r, theta)
    # r is radius of inner ring, theta is angular offset
    # Symmetry suggests theta around pi/7 or 0
    candidates = [
        (0.5, 0.0),
        (0.5, np.pi/7),
        (0.6, 0.0),
        (0.6, np.pi/7),
        (0.7, 0.0),
        (0.7, np.pi/7),
        (0.8, 0.0),
        (0.8, np.pi/7),
    ]
    
    # Phase 1: Optimize r and theta for the two-ring structure using soft-min
    # This is fast and helps find a good basin.
    
    for r_init, th_init in candidates:
        if time.time() > deadline - 30.0:
            break
            
        init_pts = _generate_two_ring(r_init, th_init)
        init_flat = init_pts.flatten()
        
        # Use L-BFGS-B on the flat coordinates, but constrained to the two-ring structure?
        # No, L-BFGS-B on 28 coordinates is better for flexibility, but might break symmetry.
        # Let's stick to optimizing r and theta first, then polish all coordinates.
        
        def neg_soft_ring(params):
            r, th = params
            if r < 0.1 or r > 0.99:
                return 1e9
            pts = _generate_two_ring(r, th)
            return _neg_soft_score(pts.flatten(), beta=20.0)
        
        try:
            res_ring = minimize(neg_soft_ring, [r_init, th_init], method='L-BFGS-B',
                                bounds=[(0.1, 0.99), (0, 2*np.pi)],
                                options={'maxiter': 100, 'ftol': 1e-12})
            
            r_opt, th_opt = res_ring.x
            pts_opt = _generate_two_ring(r_opt, th_opt)
            score_opt = _score(pts_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
                
        except Exception:
            continue
            
    # Phase 2: Full coordinate optimization using Soft-Min
    # Start from the best two-ring configuration found above.
    # If no good two-ring found, use a default.
    if best_pts is None:
        best_pts = _generate_two_ring(0.6, np.pi/7)
        
    # We will run L-BFGS-B on the full 28 coordinates.
    # This is more expensive per step but can break the ring symmetry if beneficial.
    
    current_pts = best_pts.copy()
    
    # Multiple restarts with different beta values (coarse to fine)
    betas = [10.0, 30.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
            
        current_flat = current_pts.flatten()
        
        try:
            res_full = minimize(_neg_soft_score, current_flat, args=(beta,), method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            new_pts = res_full.x.reshape(N, 2)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
                
        except Exception:
            continue
            
    # Phase 3: Exact Min-Area Polish
    # Use Nelder-Mead on the exact objective function.
    # This handles the non-differentiability of the min function.
    
    current_pts = best_pts.copy()
    
    # Run a few iterations of Nelder-Mead
    # To speed up, we can use a smaller initial simplex or just rely on NM's adaptive nature.
    
    if time.time() < deadline - 5.0:
        try:
            res_nm = minimize(_neg_score, current_pts.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            new_pts = res_nm.x.reshape(N, 2)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                
        except Exception:
            pass
            
    # Phase 4: Local Coordinate Descent / Small Perturbations
    # If time remains, try small random perturbations and keep if better.
    
    if time.time() < deadline - 2.0:
        current_pts = best_pts.copy()
        step_size = 0.01
        
        # Try a few random directions
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            
            # Generate random perturbation
            perturbation = rng.normal(0, step_size, (N, 2))
            new_pts = current_pts + perturbation
            
            # Optional: normalize to keep in unit area hull? 
            # The score is normalized by hull area, so scaling doesn't change score.
            # Translation doesn't change score.
            # So we can just check the score.
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                current_pts = new_pts.copy()
                best_pts = new_pts.copy()
            else:
                # If it didn't improve, maybe try a smaller step?
                # For simplicity, just keep the best found so far.
                pass
                
    # Final verification
    if best_pts is None:
        # Fallback to a regular 14-gon if everything failed (unlikely)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END
