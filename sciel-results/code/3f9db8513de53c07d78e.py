import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 13

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        # Check for degenerate points
        if len(points) < 3:
            return 0.0
        
        # Compute Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Compute Min Triangle Area
        # Number of triangles C(13, 3) = 286
        # Vectorized calculation
        idx = np.array(list(itertools.combinations(range(N), 3)))
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -score(points)

def generate_mixed_polygon(params):
    """
    Generate 13 points as a weighted mixture of regular polygons.
    
    params: array of shape (6,)
    - params[0]: radius of outer polygon (12-gon)
    - params[1]: radius of inner polygon (center point, 0 if we want a center)
    - params[2]: rotation angle of outer polygon
    - params[3]: rotation angle of inner polygon  
    - params[4]: weight for outer polygon points (0 to 1)
    - params[5]: additional rotation for breaking symmetry
    
    Returns: array of shape (13, 2)
    """
    try:
        r_outer = params[0]
        r_inner = params[1]
        rot_outer = params[2]
        rot_inner = params[3]
        weight_outer = params[4]
        extra_rot = params[5]
        
        points = []
        
        # Generate 12 outer points (regular 12-gon)
        angles_outer = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rot_outer + extra_rot
        pts_outer = np.column_stack((r_outer * np.cos(angles_outer), 
                                      r_outer * np.sin(angles_outer)))
        points.extend(pts_outer)
        
        # Generate 1 center/inner point
        if abs(r_inner) < 1e-6:
            # Point at center
            points.append(np.array([0.0, 0.0]))
        else:
            # Point on inner circle
            angle_inner = rot_inner + extra_rot
            points.append(np.array([r_inner * np.cos(angle_inner), 
                                    r_inner * np.sin(angle_inner)]))
        
        # Convert to array
        points_array = np.array(points)
        
        # Ensure we have exactly 13 points
        if len(points_array) != N:
            # Pad or trim if necessary
            if len(points_array) < N:
                points_array = np.vstack([points_array, np.zeros((N - len(points_array), 2))])
            else:
                points_array = points_array[:N]
        
        return points_array
    except Exception:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))

def neg_score_mixed(params):
    """
    Negative score for the mixed polygon parameterization.
    """
    try:
        points = generate_mixed_polygon(params)
        return -score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_score = 0.0
    best_points = None
    
    # First, get a baseline from regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_13 = np.column_stack((np.cos(angles), np.sin(angles)))
    baseline_score = score(init_13)
    best_score = baseline_score
    best_points = init_13.copy()
    
    # Define bounds for the mixed polygon parameters
    # r_outer: [0.5, 2.0]
    # r_inner: [-0.5, 0.5] (can be negative for phase shift)
    # rot_outer: [0, 2*pi]
    # rot_inner: [0, 2*pi]
    # weight_outer: [0, 1]
    # extra_rot: [0, 2*pi]
    bounds = [
        (0.5, 2.0),
        (-0.5, 0.5),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
        (0, 1),
        (0, 2 * np.pi)
    ]
    
    # Use differential evolution to search the parameter space
    try:
        de_result = differential_evolution(
            neg_score_mixed,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=True,
            init='latinhypercube',
            updating='immediate'
        )
        
        if de_result.fun < 0:
            de_score = -de_result.fun
            if de_score > best_score:
                best_score = de_score
                best_points = generate_mixed_polygon(de_result.x)
                
    except Exception:
        pass
    
    # If time permits, do local refinement from the best found configuration
    if time.time() < deadline - 30:
        try:
            # Convert best_points to a flat array for local optimization
            initial_flat = best_points.flatten()
            
            res = minimize(
                neg_score,
                initial_flat,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < 0:
                local_score = -res.fun
                if local_score > best_score:
                    best_score = local_score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception:
            pass
    
    # Verify final result
    if best_points is None or best_score <= 0:
        best_points = init_13
    
    return best_points
