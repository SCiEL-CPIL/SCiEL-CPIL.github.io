import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    # points shape (N, 2)
    # TRIPLE_INDICES shape (M, 3)
    
    # Extract points for all triples
    # This creates a (M, 3, 2) array
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product (pj - pi) x (pk - pi)
    # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _score_radial_pairs(r, phi):
    """
    Score for 7 pairs of points at angles k*2*pi/7.
    Outer points at radius 1.
    Inner points at radius r, angle offset by phi.
    """
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles)
    y_out = np.sin(angles)
    
    angles_in = angles + phi
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _calculate_score(points)

def _optimize_radial_pairs(time_limit):
    """
    Optimize the radial pair configuration (7-fold symmetry).
    """
    start_time = time.time()
    
    best_score = 0.0
    best_params = (0.5, 0.0)
    
    # 1. Coarse grid search for r and phi
    # r in [0.1, 0.99], phi in [0, 2*pi/7)
    r_vals = np.linspace(0.1, 0.99, 20)
    phi_vals = np.linspace(0, 2*np.pi/7, 10, endpoint=False)
    
    for r in r_vals:
        if time.time() - start_time > time_limit * 0.5:
            break
        for phi in phi_vals:
            score = _score_radial_pairs(r, phi)
            if score > best_score:
                best_score = score
                best_params = (r, phi)
                
    # 2. Local polish using Nelder-Mead
    if time.time() - start_time < time_limit * 0.8:
        x0 = np.array(best_params)
        bounds = [(0.01, 0.99), (0, 2*np.pi/7)]
        
        # We need to handle the periodic boundary for phi if using standard minimize, 
        # but Nelder-Mead doesn't support bounds directly. 
        # Since the search space is small and we started from a good grid point, 
        # we can just run it and hope it stays in range or wrap it.
        # Better: use a penalty or just trust the local nature.
        
        def obj(p):
            r, phi = p
            if r < 0.01 or r > 0.99:
                return 100.0
            # Wrap phi to [0, 2*pi/7) for consistency in scoring if needed, 
            # but _score_radial_pairs is periodic in phi with period 2*pi/7.
            return -_score_radial_pairs(r, phi)
            
        try:
            res = minimize(
                obj, 
                x0, 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 3. Generate points
    r, phi = best_params
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles)
    y_out = np.sin(angles)
    
    angles_in = angles + phi
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return best_score, points

def _coordinate_polish(points, time_limit):
    """
    Final polish of the point coordinates to maximize min triangle area / hull area.
    """
    start_time = time.time()
    
    def obj(p):
        pts = p.reshape(-1, 2)
        return -_calculate_score(pts)
        
    x0 = points.flatten()
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
        )
        
        if time.time() - start_time < time_limit:
            if -res.fun > -obj(x0):
                return res.x.reshape(-1, 2)
    except Exception:
        pass
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Symmetric Radial Pairs (7-fold symmetry)
    # This is the specific family suggested in the directive.
    score1, points1 = _optimize_radial_pairs(40.0)
    if score1 > best_score:
        best_score = score1
        best_points = points1
        
    # Strategy 2: General Ring Optimization (Fallback/Comparison)
    # Keep the original logic but with reduced time if Strategy 1 is strong.
    # We will only run this if we have time left.
    if time.time() < deadline - 30:
        candidates = [
            (7, 7),
            (8, 6),
            (6, 8),
        ]
        
        for n_out, n_in in candidates:
            if time.time() > deadline - 15:
                break
            
            # Re-use the original _optimize_ring logic but simplified
            # For brevity and speed, we'll just do a quick DE for 7-7 and 8-6
            if (n_out, n_in) == (7, 7):
                t_limit = 20.0
            else:
                t_limit = 10.0
                
            # We can call a simplified version of the original optimization
            # or just skip if we trust the symmetric model.
            # Let's implement a quick check for 8-6 to see if it beats 7-7 symmetric.
            
            # To keep code concise and focused on the directive, we'll stick to the symmetric model 
            # and the coordinate polish. The original program's general ring search 
            # is slower and may not find better than the symmetric optimum if it exists.
            # However, to be safe, let's do a very quick DE for 8-6.
            
            def _score_ring_params_quick(n_out, n_in, params):
                r, theta = params
                if r < 0 or r > 0.999:
                    return 0.0
                angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                x_out = np.cos(angles_out)
                y_out = np.sin(angles_out)
                angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                x_in = r * np.cos(angles_in)
                y_in = r * np.sin(angles_in)
                points = np.vstack([
                    np.column_stack((x_out, y_out)),
                    np.column_stack((x_in, y_in))
                ])
                return _calculate_score(points)
                
            try:
                res = differential_evolution(
                    lambda p: -_score_ring_params_quick(n_out, n_in, p),
                    [(0.0, 0.99), (0.0, 2*np.pi)],
                    seed=42,
                    maxiter=20,
                    popsize=10,
                    tol=1e-4,
                    polish=False
                )
                if res.success:
                    score2 = -res.fun
                    if score2 > best_score:
                        best_score = score2
                        r, theta = res.x
                        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                        x_out = np.cos(angles_out)
                        y_out = np.sin(angles_out)
                        angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                        x_in = r * np.cos(angles_in)
                        y_in = r * np.sin(angles_in)
                        best_points = np.vstack([
                            np.column_stack((x_out, y_out)),
                            np.column_stack((x_in, y_in))
                        ])
            except Exception:
                pass

    # Final Coordinate Polish
    if best_points is not None and time.time() < deadline - 5:
        best_points = _coordinate_polish(best_points, deadline - time.time() - 2)
        final_score = _calculate_score(best_points)
        if final_score > best_score:
            best_score = final_score

    # Fallback
    if best_points is None:
        # Regular 14-gon
        angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
