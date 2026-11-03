import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
TIME_LIMIT = 115.0 # Safety margin

# Pre-compute indices for all triples
TRIPLE_INDICES = list(combinations(range(N), 3))
TRIPLE_ARRAY = np.array(TRIPLE_INDICES) # shape (286, 3)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) != N:
        return -1e10
        
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return -1e10
    except Exception:
        return -1e10

    # Calculate areas of all triangles
    # points[TRIPLE_ARRAY[:, 0]] gives shape (286, 2)
    p0 = points[TRIPLE_ARRAY[:, 0]]
    p1 = points[TRIPLE_ARRAY[:, 1]]
    p2 = points[TRIPLE_ARRAY[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return -1e10
        
    return min_area / hull_area

def _neg_score(points_flat):
    """
    Objective function for scipy.optimize.minimize.
    Takes a flat array of size 2*N, reshapes to (N, 2), and returns negative score.
    """
    points = points_flat.reshape(N, 2)
    return -_score(points)

def _generate_initialization(method='grid', seed=0):
    """
    Generate initial points.
    """
    rng = np.random.default_rng(seed)
    
    if method == 'grid':
        # 3x5 grid or similar, slightly perturbed
        # 13 points. 3 rows of 4 and 1 row of 1? Or 4x4 minus 3?
        # Let's try a 4x4 grid and pick 13 points
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.9, 4)
        grid_points = []
        for x in xs:
            for y in ys:
                grid_points.append([x, y])
        # Shuffle and take first 13
        rng.shuffle(grid_points)
        points = np.array(grid_points[:13])
        # Add small noise to break symmetry
        points += rng.normal(0, 0.01, points.shape)
        return points
        
    elif method == 'halton':
        def halton(index, base):
            f = 1.0
            result = 0.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result
        
        points = []
        for i in range(1, N + 1):
            x = halton(i, 2)
            y = halton(i, 3)
            points.append([x, y])
        points = np.array(points)
        points += rng.normal(0, 0.01, points.shape)
        return points
        
    elif method == 'random':
        return rng.random((N, 2)) * 0.8 + 0.1

    else:
        return rng.random((N, 2)) * 0.8 + 0.1

def _optimize_from(points, max_iter=1000, timeout_check=None):
    """
    Run Nelder-Mead optimization starting from points.
    """
    initial_flat = points.flatten()
    
    # Options for Nelder-Mead
    options = {
        'maxiter': max_iter,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'disp': False
    }
    
    try:
        res = minimize(_neg_score, initial_flat, method='Nelder-Mead', options=options)
        if res.success or res.fun < 0: # res.fun is neg score, so < 0 means positive score
            return res.x.reshape(N, 2), -res.fun
        else:
            # If it failed, maybe return the best point found? 
            # Nelder-Mead doesn't easily expose intermediate bests in standard API without callbacks.
            # But usually it improves.
            return points, _score(points)
    except Exception as e:
        # print(e)
        return points, _score(points)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -1e10
    
    # List of initializations
    inits = []
    
    # 1. Grid-based
    inits.append(_generate_initialization('grid', seed=1))
    inits.append(_generate_initialization('grid', seed=2))
    
    # 2. Halton
    inits.append(_generate_initialization('halton', seed=1))
    inits.append(_generate_initialization('halton', seed=2))
    
    # 3. Random
    inits.append(_generate_initialization('random', seed=1))
    inits.append(_generate_initialization('random', seed=2))
    
    # 4. Regular polygon (points on circle) - might be bad but good start for symmetry breaking?
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    poly_points = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.4 + 0.5
    inits.append(poly_points)
    
    # 5. Hexagonal lattice subset
    # Generate a hex lattice in [0,1]x[0,1] and pick closest to ideal spots
    # Simpler: just use a structured 3x5 grid
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 3)
    hex_like = []
    for i, y in enumerate(ys):
        offset = 0 if i % 2 == 0 else (xs[1]-xs[0])/2
        for x in xs:
            hex_like.append([x+offset, y])
    # Take first 13
    hex_like = np.array(hex_like[:13])
    inits.append(hex_like)
    
    # Optimization Loop
    for i, init_points in enumerate(inits):
        if time.time() > deadline:
            break
            
        current_points = init_points
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Run local optimization
        opt_points, opt_score = _optimize_from(current_points, max_iter=500)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
            
        # Iterated Local Search: Perturb the best found so far and re-optimize
        # We do this a few times if time permits
        for restart in range(3):
            if time.time() > deadline:
                break
                
            # Perturb the current best points
            # Use a step size that is significant but not too large
            step_size = 0.1
            rng = np.random.default_rng(hash(tuple(best_points[0])) + restart)
            perturbed_points = best_points + rng.normal(0, step_size, best_points.shape)
            # Keep points within bounds roughly, though not strictly necessary for ratio
            # But keeping them in a bounded box helps numerical stability
            perturbed_points = np.clip(perturbed_points, 0.0, 1.0)
            
            opt_points2, opt_score2 = _optimize_from(perturbed_points, max_iter=500)
            
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_points2.copy()
                
    # Final check: if best_points is None, return a default
    if best_points is None:
        best_points = inits[0]
        
    # Optional: One final polish on the best points
    if time.time() < deadline - 10:
        final_points, final_score = _optimize_from(best_points, max_iter=1000)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
            
    return best_points
