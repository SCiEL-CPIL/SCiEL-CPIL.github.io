"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _negative_score(pts_flat):
    """
    Objective function for scipy.optimize.minimize.
    Takes a flat array of points, reshapes, computes score, returns negative score.
    """
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s

def _initial_construction_regular():
    """14 points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _initial_construction_grid():
    """4x4 grid, slightly perturbed to avoid degeneracy."""
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    # Add small noise to break symmetry
    rng = np.random.default_rng(123)
    pts += rng.normal(0, 0.01, pts.shape)
    return pts

def _initial_construction_rings():
    """2 rings: 7 outer, 7 inner."""
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    r_outer = 1.0
    r_inner = 0.5
    
    outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _initial_construction_centered():
    """1 center, 6 inner, 7 outer."""
    center = np.array([[0.0, 0.0]])
    
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r_inner = 0.5
    inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    r_outer = 1.0
    outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    
    pts = np.vstack([center, inner_pts, outer_pts])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # List of initial configurations
    inits = [
        _initial_construction_regular(),
        _initial_construction_grid(),
        _initial_construction_rings(),
        _initial_construction_centered()
    ]
    
    best_score = -float('inf')
    best_pts = None
    
    # Nelder-Mead parameters
    # maxiter is set high, but we rely on time checks
    # adaptive=True helps with scaling
    options = {
        'maxiter': 500, 
        'adaptive': True,
        'xatol': 1e-6,
        'fatol': 1e-6
    }
    
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Use Nelder-Mead
            res = minimize(
                _negative_score, 
                x0, 
                method='Nelder-Mead', 
                options=options
            )
            
            if res.success or res.fun < best_score:
                new_pts = res.x.reshape(-1, 2)
                new_score = -res.fun
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
                    
        except Exception:
            # If optimization fails, try a few random restarts from this init
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
        # If we have time, do a second pass with different options or perturbed start?
        # For now, let's just move to the next init.
        
    # If we still have time, we can run a final polish on the best found
    if best_pts is not None and time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        try:
            # Tighter tolerance for final polish
            res_final = minimize(
                _negative_score, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_final.fun < best_score:
                best_score = -res_final.fun
                best_pts = res_final.x.reshape(-1, 2)
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
