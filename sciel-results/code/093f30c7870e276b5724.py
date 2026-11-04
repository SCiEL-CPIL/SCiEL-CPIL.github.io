"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    # Precompute for efficiency
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xi_j = xi - pts[j, 0]
            yi_j = yi - pts[j, 1]
            for k in range(j + 1, n):
                cross = xi_j * (pts[k, 1] - yi) - yi_j * (pts[k, 0] - xi)
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the point set."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _score_params_symmetric(params):
    """
    Compute score for a symmetric configuration of two concentric heptagons.
    params: [r, theta]
    r: radius of inner heptagon (outer is 1.0)
    theta: angular offset of inner heptagon
    """
    r, theta = params
    # Ensure r is positive and less than 1
    if r <= 0 or r >= 1:
        return -1e9 # Penalty
    
    # Outer heptagon angles
    angles_outer = 2 * np.pi * np.arange(7) / 7
    pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    # Inner heptagon angles
    angles_inner = 2 * np.pi * np.arange(7) / 7 + theta
    pts_inner = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([pts_outer, pts_inner])
    
    return -_score(pts) # Negate for minimization

def _optimize_symmetric(deadline):
    """Optimize the symmetric 7-7 configuration."""
    best_score = 0.0
    best_pts = None
    
    # Initial guesses for r and theta
    # theta is in [0, 2*pi/7) due to symmetry
    max_theta = 2 * np.pi / 7
    guesses = [
        [0.5, max_theta / 2],
        [0.6, max_theta / 2],
        [0.4, max_theta / 2],
        [0.5, 0.1 * max_theta],
        [0.5, 0.9 * max_theta],
        [0.7, max_theta / 2],
        [0.3, max_theta / 2]
    ]
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_score_params_symmetric, guess, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 1000})
            
            if res.success or (hasattr(res, 'fun') and res.fun < 0):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    r, theta = res.x
                    # Reconstruct points
                    angles_outer = 2 * np.pi * np.arange(7) / 7
                    pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
                    angles_inner = 2 * np.pi * np.arange(7) / 7 + theta
                    pts_inner = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
                    best_pts = np.vstack([pts_outer, pts_inner])
        except Exception:
            continue
            
    return best_pts, best_score

def construct():
    start = time.time()
    deadline = start + 110.0  # Leave 10s margin
    
    # 1. Try symmetric 7-7 configuration
    pts_sym, score_sym = _optimize_symmetric(deadline)
    
    best_pts = pts_sym
    best_score = score_sym
    
    # 2. If we have time, try a general local search starting from the symmetric solution
    # This allows breaking symmetry if it helps.
    if best_pts is not None and time.time() < deadline - 20.0:
        # Use a simple coordinate ascent or random perturbation local search
        # but constrained to be fast.
        current_pts = best_pts.copy()
        current_score = best_score
        step = 0.05
        no_improve_count = 0
        
        while time.time() < deadline - 5.0 and no_improve_count < 200:
            # Pick a random point and move it
            idx = np.random.randint(len(current_pts))
            direction = np.random.randn(2)
            direction /= (np.linalg.norm(direction) + 1e-12)
            
            new_pt = current_pts[idx].copy()
            new_pt += step * direction
            
            old_pt = current_pts[idx].copy()
            current_pts[idx] = new_pt
            
            new_score = _score(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                current_pts[idx] = old_pt
                no_improve_count += 1
                # If stuck, shrink step or try larger random jump
                if no_improve_count % 50 == 0:
                    step *= 0.8
                    if step < 1e-4:
                        break
                        
    # 3. Fallback if symmetric failed
    if best_pts is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
# EVOLVE-BLOCK-END
