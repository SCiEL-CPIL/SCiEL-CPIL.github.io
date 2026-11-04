import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _objective_constrained(params, beta=50.0):
    # params contains coordinates for P[2] to P[12]
    # P[0] = (0,0), P[1] = (1,0)
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:, :] = params.reshape(N-2, 2)
    
    # Check for duplicate points or very small distances
    # If any point is too close to another, penalty
    # Simple check: min distance > 1e-5
    # We can just let the area calculation handle it, but 0/0 is bad.
    
    # Calculate triangle areas
    areas = np.zeros(286)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                v1 = P[j] - P[i]
                v2 = P[k] - P[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas[idx] = 0.5 * abs(cross)
                idx += 1
    
    # Hull Area
    try:
        hull = ConvexHull(P)
        hull_area = hull.volume
    except Exception:
        return 1e10 # Large penalty for degenerate hull
        
    if hull_area < 1e-10:
        return 1e10

    # Soft min
    # Use shift for stability
    max_a = np.max(areas)
    # If max_a is 0, all points collinear
    if max_a < 1e-10:
        return 1e10
        
    exp_terms = np.exp(-beta * (areas - max_a))
    sum_exp = np.sum(exp_terms)
    soft_min_area = max_a - (1.0 / beta) * np.log(sum_exp)
    
    score = soft_min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    timeout = 110.0
    
    best_score = -np.inf
    best_params = None
    
    # Initial guess: Regular 13-gon
    # Need to map regular 13-gon to the constrained frame.
    # Generate regular 13-gon on unit circle.
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    P_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Rotate/Translate/Scale so P[0]=(0,0) and P[1]=(1,0)?
    # The regular polygon has vertices on the circle.
    # We can just pick any two vertices to be P0 and P1?
    # No, the relative order matters for the convex hull if we don't sort.
    # But ConvexHull sorts internally.
    # The constrained parameters just define the set of points.
    # So we can take the regular polygon, and set:
    # p0 = P_reg[0]
    # p1 = P_reg[1]
    # Vector d = p1 - p0.
    # Scale and rotate the whole polygon so that p0 becomes (0,0) and p1 becomes (1,0).
    
    p0 = P_reg[0]
    p1 = P_reg[1]
    d = p1 - p0
    scale = np.linalg.norm(d)
    # Rotation matrix to align d with x-axis
    theta = np.arctan2(d[1], d[0])
    R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    
    # Transform all points
    P_trans = (P_reg - p0) @ R.T / scale
    
    # Now P_trans[0] is (0,0) and P_trans[1] is (1,0)
    # Extract parameters for P[2:]
    p_init = P_trans[2:, :].flatten()
    
    initial_guesses = [p_init]
    
    # Add some perturbations
    rng = np.random.default_rng(42)
    for i in range(5):
        perturbation = rng.normal(0, 0.1, (N-2, 2))
        p_init_pert = p_init + perturbation.flatten()
        initial_guesses.append(p_init_pert)
        
    for p0 in initial_guesses:
        if time.time() - start_time > timeout:
            break
            
        try:
            # Phase 1: Lower beta for smoothness
            res1 = minimize(_objective_constrained, p0, args=(10.0,), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
            
            # Phase 2: Higher beta for precision
            if res1.success:
                res2 = minimize(_objective_constrained, res1.x, args=(100.0,), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
                final_res = res2
            else:
                final_res = res1
                
            if final_res.fun < -best_score: # Remember objective is -score
                best_score = -final_res.fun
                best_params = final_res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to regular polygon
        return P_trans
        
    # Reconstruct points
    P_final = np.zeros((N, 2))
    P_final[0] = [0.0, 0.0]
    P_final[1] = [1.0, 0.0]
    P_final[2:, :] = best_params.reshape(N-2, 2)
    
    return P_final
