import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

# Precompute indices for all combinations of 3 points
IDX_I, IDX_J, IDX_K = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    # Since points are in convex position, the hull is the polygon formed by the points in order
    # Shoelace formula
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    return area

def _score_from_radii(radii, angles):
    """
    Compute score given radii and fixed/sorted angles.
    Points are at (r_i * cos(a_i), r_i * sin(a_i)).
    """
    # Ensure points are in sorted angular order for hull area calculation
    # The input radii are assumed to correspond to sorted angles
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    
    min_area = _compute_min_triangle_area(pts)
    return min_area / hull_area

def _objective_r(radii, angles):
    return -_score_from_radii(radii, angles)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_points = None
    
    # Base angles: Regular polygon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # We optimize the radii. 
    # Strategy: 
    # 1. Start with equal radii (regular polygon).
    # 2. Use Nelder-Mead to adjust radii to maximize min triangle area / hull area.
    # 3. Try different angular offsets (rotations) and slight angle perturbations?
    #    Actually, rotation is symmetric, so fixed angles are fine.
    #    However, the optimal convex configuration might not have equally spaced angles.
    #    But for n=14, the regular polygon is a strong local minimum. 
    #    To escape, we can optimize both angles and radii? 
    #    The directive suggests "convex position parameterization (e.g., sorted angles)".
    #    Let's stick to optimizing radii on fixed regular angles first, 
    #    then if we have time, try optimizing angles as well.
    
    # Optimization 1: Fixed Angles (Regular Polygon Spacing), Varying Radii
    initial_radii = np.ones(N)
    
    try:
        # Nelder-Mead on 14 variables
        res = minimize(
            _objective_r, 
            initial_radii, 
            args=(base_angles,), 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        opt_radii = res.x
        # Ensure radii are positive
        opt_radii = np.maximum(opt_radii, 0.1)
        
        pts = np.column_stack((opt_radii * np.cos(base_angles), opt_radii * np.sin(base_angles)))
        score = _score(pts) if hasattr(_score, '__call__') else _score_from_radii(opt_radii, base_angles)
        # Re-calculate score properly using the helper to be safe
        score = _score_from_radii(opt_radii, base_angles)
        
        if score > best_score:
            best_score = score
            best_points = pts
            
    except Exception:
        pass

    # Optimization 2: Allow angles to vary slightly? 
    # If we allow angles to vary, we must maintain sorting.
    # Let's try a combined optimization of angles and radii, but initialized near regular.
    # This is 28 variables again, but constrained to convex position.
    # Actually, if we just use the best points from Optimization 1 as a seed for a general 
    # Cartesian optimizer, we might find a better non-regular convex shape.
    
    if best_points is not None and time.time() < deadline - 20.0:
        # Refine the convex solution using a Cartesian optimizer, 
        # but constrain it to stay "close" to convex? 
        # Or just use the convex result as a good starting point for the previous Cartesian method?
        # The previous method scored 34.4. The convex method might find a specific structure.
        
        # Let's try to optimize the angles as well, keeping points in convex order.
        # Variables: [theta_1, ..., theta_14, r_1, ..., r_14]
        # Constraints: theta_i < theta_{i+1} < theta_i + 2pi
        # This is tricky with standard minimize. 
        # Alternative: Optimize deviations from regular angles.
        
        delta_angles = np.zeros(N)
        # Normalize angles to [0, 1) range for optimization stability? 
        # Let's stick to the radii-only optimization for now as it's the core of the directive's suggestion 
        # to reduce dimensionality. The "jump" might come from escaping the regular polygon 
        # local minimum by adjusting radii, which the 28-dim Cartesian optimizer might have missed 
        # or struggled with due to the flat manifold of rotations/translations.
        
        # Let's try a few different angular shifts to break symmetry? 
        # No, the score is rotationally invariant. 
        # But the *relative* angles matter. 
        # What if the optimal shape is not a regular 14-gon but a "squashed" one?
        
        # Try optimizing radii with a different initialization: 
        # e.g., alternating radii (star-like but convex)
        init_radii_2 = np.ones(N)
        init_radii_2[::2] = 1.2
        init_radii_2[1::2] = 0.8
        
        res2 = minimize(
            _objective_r, 
            init_radii_2, 
            args=(base_angles,), 
            method='Nelder-Mead', 
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        opt_radii2 = np.maximum(res2.x, 0.1)
        pts2 = np.column_stack((opt_radii2 * np.cos(base_angles), opt_radii2 * np.sin(base_angles)))
        score2 = _score_from_radii(opt_radii2, base_angles)
        
        if score2 > best_score:
            best_score = score2
            best_points = pts2

    # Fallback: If convex optimization didn't beat the known best, return the regular polygon 
    # or the best found. The regular polygon score is approx 1/14 * something?
    # For a regular n-gon, min triangle area / hull area is a known value.
    # Score 34.43 suggests the previous Cartesian method found a configuration significantly better 
    # than the regular polygon (if the regular polygon is lower).
    
    # Wait, if the previous program scored 34.43 and it used Cartesian coords, 
    # and the regular polygon is a specific case, did it beat the regular polygon?
    # If the regular polygon is the global optimum for convex sets, the score should be fixed.
    # If 34.43 > Regular Polygon Score, then the optimum is NOT the regular polygon.
    # The convex parameterization allows us to search *only* convex sets.
    # If the global optimum is non-convex (points inside hull), the convex parameterization 
    # will fail to find it. However, for "maximize min triangle area", it is generally 
    # believed that points should be on the boundary of the convex hull (Tammes problem analog).
    # If the previous score 34.43 came from a non-convex set, this new method might score lower.
    # But the task says "Place 14 points in a planar convex region". 
    # It does NOT say the points must be in convex position (on the hull). 
    # However, for maximizing min distance/area, boundary points are usually optimal.
    
    # To be safe, if the convex search doesn't beat a baseline, we should maybe fall back to 
    # the previous logic? No, we must output a distinct program. 
    # The directive explicitly asks for the convex position parameterization.
    
    if best_points is None:
        # Fallback to regular polygon
        best_points = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        
    return best_points

# Helper for the fallback check in the code above
def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area
