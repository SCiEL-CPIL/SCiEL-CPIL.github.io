import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
N_OUTER = 7
N_INNER = 6

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _decode_params(params):
    """
    Decode optimization parameters into points.
    params structure:
    [0]: outer_radius (global scale)
    [1]: inner_radius_ratio (relative to outer)
    [2]: rotation_offset (angle offset for inner shell relative to outer)
    [3:16]: angular perturbations for outer 7 points
    [16:22]: angular perturbations for inner 6 points
    [22:29]: radial perturbations for all 13 points (relative to their shell radius)
    """
    R_out = params[0]
    R_in_ratio = params[1]
    rot_offset = params[2]
    
    R_in = R_out * R_in_ratio
    
    # Base angles for regular polygons
    angles_out_base = 2 * np.pi * np.arange(N_OUTER) / N_OUTER
    angles_in_base = 2 * np.pi * np.arange(N_INNER) / N_INNER + rot_offset
    
    # Perturbations
    ang_pert_out = params[3:3+N_OUTER]
    ang_pert_in = params[3+N_OUTER:3+N_OUTER+N_INNER]
    
    # Radial perturbations (applied to all N points)
    rad_pert = params[3+N_OUTER+N_INNER:3+N_OUTER+N_INNER+N]
    
    # Construct outer points
    angles_out = angles_out_base + ang_pert_out
    radii_out = R_out * (1.0 + rad_pert[:N_OUTER])
    
    pts_out = np.column_stack((radii_out * np.cos(angles_out), radii_out * np.sin(angles_out)))
    
    # Construct inner points
    angles_in = angles_in_base + ang_pert_in
    radii_in = R_in * (1.0 + rad_pert[N_OUTER:N_OUTER+N_INNER])
    
    pts_in = np.column_stack((radii_in * np.cos(angles_in), radii_in * np.sin(angles_in)))
    
    return np.vstack([pts_out, pts_in])

def _objective(params):
    """
    Objective function for optimization.
    Returns negative score to maximize.
    """
    try:
        pts = _decode_params(params)
        score = _exact_score(pts)
        return -score
    except Exception:
        return 1e10

def construct():
    n = N
    t0 = time.time()
    deadline = 140.0 # Safety margin from 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Define bounds for the 29 parameters
    # [R_out, R_in_ratio, rot_offset, ang_pert_out (7), ang_pert_in (6), rad_pert (13)]
    
    # R_out: 0.5 to 2.0 (scale doesn't matter for ratio, but helps optimizer)
    # R_in_ratio: 0.1 to 1.0 (inner should be smaller)
    # rot_offset: 0 to 2*pi
    # ang_pert: -pi/12 to pi/12 (small perturbations to maintain structure)
    # rad_pert: -0.3 to 0.3 (30% radial variation allowed)
    
    bounds = [
        (0.5, 2.0),      # R_out
        (0.1, 0.9),      # R_in_ratio
        (0.0, 2*np.pi),  # rot_offset
    ]
    
    # 7 outer angular perturbations
    bounds += [(-np.pi/10, np.pi/10)] * N_OUTER
    # 6 inner angular perturbations
    bounds += [(-np.pi/10, np.pi/10)] * N_INNER
    # 13 radial perturbations
    bounds += [(-0.4, 0.4)] * N
    
    # Initial guess: Regular 7-gon and Regular 6-gon, concentric, no perturbations
    # R_out=1, R_in=0.5, rot=0, all perts=0
    x0 = np.zeros(29)
    x0[0] = 1.0
    x0[1] = 0.5
    x0[2] = 0.0
    
    # Strategy 1: Differential Evolution (Global Search)
    # DE is slow but robust. We limit popsize and maxiter to fit in time.
    if time.time() - t0 < deadline - 30:
        try:
            # Use 'best1bin' mutation, small population for speed
            # popsize=20, maxiter=50 should be fast enough for 29 dims
            res_de = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=40, 
                popsize=25, 
                tol=1e-8, 
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=True, # Let DE do a final local polish
                init='sobol' # Good initial spread
            )
            
            if res_de.success or res_de.fun < best_score:
                best_params = res_de.x
                best_points = _decode_params(best_params)
                best_score = -res_de.fun
        except Exception as e:
            pass

    # Strategy 2: Multi-start Local Refinement
    # If DE failed or we have time, run local refinements from diverse starts.
    # We generate several random perturbations around the regular polygon structure.
    
    if best_points is not None:
        # Polish the best DE result
        try:
            res_local = minimize(_objective, best_params, method='Nelder-Mead', 
                                 options={'maxiter': 200, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True})
            if res_local.fun < best_score:
                best_score = -res_local.fun
                best_params = res_local.x
                best_points = _decode_params(best_params)
        except Exception:
            pass

    # Try a few random restarts if time permits
    if time.time() - t0 < deadline - 10:
        for restart in range(5):
            if time.time() - t0 > deadline - 5:
                break
            
            # Create a random perturbation around the ideal symmetric start
            x0_rand = x0.copy()
            # Randomize rotation
            x0_rand[2] = np.random.uniform(0, 2*np.pi)
            # Randomize radius ratio slightly
            x0_rand[1] = np.random.uniform(0.4, 0.7)
            # Add small random perturbations
            x0_rand[3:] += np.random.uniform(-0.1, 0.1, size=26)
            
            try:
                res_local = minimize(_objective, x0_rand, method='Nelder-Mead', 
                                     options={'maxiter': 100, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True})
                if res_local.fun < best_score:
                    best_score = -res_local.fun
                    best_params = res_local.x
                    best_points = _decode_params(best_params)
            except Exception:
                pass

    # Fallback if everything failed
    if best_points is None:
        # Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
