import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14

# Precompute indices for all triangles
_TRIPLES = np.array(list(itertools.combinations(range(N), 3)))

def calc_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
    
    # Vectorized triangle area calculation
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    score = calc_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = -1.0
    best_points = None
    
    # Initial guesses
    # 1. Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Elliptical Regular Polygon (stretch x)
    init2 = np.column_stack([1.2*np.cos(angles), np.sin(angles)])
    
    # 3. Elliptical Regular Polygon (stretch y)
    init3 = np.column_stack([np.cos(angles), 1.2*np.sin(angles)])
    
    # 4. Random perturbation of regular polygon
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.1, (N, 2))
    init4 = init1 + noise
    
    inits = [init1, init2, init3, init4]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        try:
            # Use Nelder-Mead for non-smooth objective
            res = scipy.optimize.minimize(
                neg_score, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                    'maxfev': 5000
                }
            )
            
            if time.time() > deadline:
                break
                
            current_points = res.x.reshape(N, 2)
            current_score = calc_score(current_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                
        except Exception:
            continue
            
    # If best_points is still None, return a regular polygon
    if best_points is None:
        best_points = init1
        
    return best_points
