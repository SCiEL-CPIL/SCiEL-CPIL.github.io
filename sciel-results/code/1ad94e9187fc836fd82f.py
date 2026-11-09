import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _get_all_triangle_areas(points):
    """
    Calculate areas of all C(N,3) triangles formed by the points.
    Returns a numpy array of areas.
    """
    n = len(points)
    # Precompute indices for speed
    indices = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    
    # Vectorized cross product calculation
    # Area = 0.5 * | (Pj - Pi) x (Pk - Pi) |
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert parameter vector (14 angular offsets, 14 radial offsets) to Cartesian points.
    Base configuration: Regular 14-gon with radius 1.
    params[0:14] are angular offsets (radians) from the regular angles.
    params[14:28] are radial offsets from the base radius 1.
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radius = 1.0
    
    angles = base_angles + params[:N]
    radii = base_radius + params[N:]
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack([x, y])

def _objective_smooth(params, tau=50.0):
    """
    Smooth surrogate for the negative min triangle area, normalized by hull area.
    Minimize this function.
    """
    points = _params_to_points(params)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6  # Penalty for degenerate hull
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate areas
    areas = _get_all_triangle_areas(points)
    
    # 4. Smooth min approximation (log-sum-exp for soft-min)
    # soft_min(x) = -1/tau * log(sum(exp(-tau * x_i)))
    # We want to maximize soft_min, so minimize -soft_min
    tau_eff = min(tau, 100.0)  # Cap tau for numerical stability
    shifted = areas - np.min(areas)
    sum_exp = np.sum(np.exp(-tau_eff * shifted))
    
    soft_min_val = -np.min(areas) - (1.0 / tau_eff) * np.log(sum_exp)
    
    return soft_min_val

def _objective_exact(params):
    """
    Exact objective: -min_triangle_area, assuming points are normalized.
    """
    points = _params_to_points(params)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate min area
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return -min_area

def _generate_initial_params():
    """
    Generate diverse initial parameter vectors (angular and radial perturbations).
    """
    candidates = []
    
    # 1. Zero perturbations (Regular polygon)
    candidates.append(np.zeros(2 * N))
    
    # 2. Small random perturbations with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(30):
        # Angular offsets: keep small to maintain convexity
        # Max safe offset is roughly pi/N / 2 to maintain ordering
        ang_off = rng.randn(N) * 0.05
        rad_off = rng.randn(N) * 0.08
        params = np.concatenate([ang_off, rad_off])
        candidates.append(params)
        
    # 3. Structured perturbations
    # Alternating radius (14-gon with alternating in/out)
    rad_alt = np.array([0.08 if i % 2 == 0 else -0.08 for i in range(N)])
    candidates.append(np.concatenate([np.zeros(N), rad_alt]))
    
    # Sine wave radius modulation (period 2)
    rad_sine = 0.1 * np.sin(2 * np.pi * 2 * np.arange(N) / N)
    candidates.append(np.concatenate([np.zeros(N), rad_sine]))
    
    # Sine wave radius modulation (period 3)
    rad_sine3 = 0.1 * np.sin(2 * np.pi * 3 * np.arange(N) / N)
    candidates.append(np.concatenate([np.zeros(N), rad_sine3]))
    
    # Angular sine modulation
    ang_sine = 0.05 * np.sin(2 * np.pi * 2 * np.arange(N) / N)
    candidates.append(np.concatenate([ang_sine, np.zeros(N)]))
    
    # Combined small angular + radial sine
    ang_s = 0.03 * np.sin(2 * np.pi * 3 * np.arange(N) / N)
    rad_s = 0.05 * np.sin(2 * np.pi * 2 * np.arange(N) / N)
    candidates.append(np.concatenate([ang_s, rad_s]))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s limit
    
    best_score = -1.0
    best_params = None
    
    # Get initial candidates
    inits = _generate_initial_params()
    
    # Optimization Strategy:
    # 1. For each candidate, run local optimization using Powell on the smooth objective.
    # 2. Then polish with Nelder-Mead on the exact objective.
    # 3. Use remaining time for differential evolution.
    
    for x0 in inits:
        if time.time() > deadline - 30.0:  # Leave time for DE
            break
        
        try:
            # Phase 1: Smooth local optimization with Powell
            res1 = minimize(_objective_smooth, x0, method='Powell', 
                            options={'maxiter': 200, 'xtol': 1e-7, 'ftol': 1e-7, 'adaptive': True})
            
            if time.time() > deadline - 30.0:
                break
                
            p1 = res1.x
            
            # Phase 2: Exact local optimization (Polish) with Nelder-Mead
            res2 = minimize(_objective_exact, p1, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            
            if time.time() > deadline - 30.0:
                break
                
            p2 = res2.x
            
            # Calculate score
            points = _params_to_points(p2)
            points = points - np.mean(points, axis=0)
            hull_area = _hull_area(points)
            if hull_area > 1e-10:
                points = points / np.sqrt(hull_area)
            
            s = _score(points)
            
            if s > best_score:
                best_score = s
                best_params = p2.copy()
                
        except Exception:
            pass
    
    # If we have time, try a broader search using Differential Evolution
    time_remaining = deadline - time.time()
    if time_remaining > 25.0:
        try:
            # Tighter bounds to maintain convexity
            ang_bounds = [(-np.pi/N * 0.8, np.pi/N * 0.8)] * N
            rad_bounds = [(-0.3, 0.3)] * N
            bounds = ang_bounds + rad_bounds
            
            # Scale population size and iterations based on available time
            max_iter = max(20, int((time_remaining - 10.0) / 2.0))
            popsize = 15
            
            res_de = differential_evolution(_objective_exact, bounds, 
                                            maxiter=max_iter, 
                                            popsize=popsize, 
                                            tol=1e-5, 
                                            seed=42,
                                            polish=True,
                                            init='sobol')
            
            if time.time() < deadline:
                p_de = res_de.x
                points = _params_to_points(p_de)
                points = points - np.mean(points, axis=0)
                hull_area = _hull_area(points)
                if hull_area > 1e-10:
                    points = points / np.sqrt(hull_area)
                
                s = _score(points)
                
                if s > best_score:
                    best_score = s
                    best_params = p_de.copy()
                    
        except Exception:
            pass
    
    # Additional polish: if we have time, do one more Nelder-Mead pass on the best
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res_final = minimize(_objective_exact, best_params, method='Nelder-Mead', 
                                 options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            p_final = res_final.x
            points = _params_to_points(p_final)
            points = points - np.mean(points, axis=0)
            hull_area = _hull_area(points)
            if hull_area > 1e-10:
                points = points / np.sqrt(hull_area)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_params = p_final.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback to regular polygon
        best_params = np.zeros(2 * N)

    # Convert best params to points
    best_points = _params_to_points(best_params)
    best_points = best_points - np.mean(best_points, axis=0)
    hull_area = _hull_area(best_points)
    if hull_area > 1e-10:
        best_points = best_points / np.sqrt(hull_area)

    return best_points

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points shape: {pts.shape}")
