import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def fourier_to_points(params):
    """
    params: array of length 4*K.
    Structure: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
    a_k, b_k for angular perturbation.
    c_k, d_k for radial perturbation.
    """
    K = 5
    n = 20
    # Base angles
    theta_base = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Initialize perturbations
    d_theta = np.zeros(n)
    d_r = np.zeros(n)
    
    for k in range(1, K+1):
        idx = (k-1)*4
        a_k = params[idx]
        b_k = params[idx+1]
        c_k = params[idx+2]
        d_k = params[idx+3]
        
        d_theta += a_k * np.sin(k * theta_base) + b_k * np.cos(k * theta_base)
        d_r += c_k * np.sin(k * theta_base) + d_k * np.cos(k * theta_base)
        
    theta = theta_base + d_theta
    r = 1.0 + d_r
    
    # Ensure r is positive
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist < 1e-6:
        # Fallback to regular polygon if degenerate
        return np.column_stack((np.cos(theta_base), np.sin(theta_base)))
    
    points = points / mean_dist
    
    return points

def vertex_score(v_idx, points):
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
        
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
        
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def min_vertex_score(params):
    points = fourier_to_points(params)
    scores = np.array([vertex_score(i, points) for i in range(20)])
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    K = 5
    num_params = 4 * K
    
    # Bounds for Fourier coefficients
    bounds = []
    for k in range(1, K+1):
        scale = 1.0 / k
        # Angular
        bounds.append((-0.3 * scale, 0.3 * scale))
        bounds.append((-0.3 * scale, 0.3 * scale))
        # Radial
        bounds.append((-0.5 * scale, 0.5 * scale))
        bounds.append((-0.5 * scale, 0.5 * scale))
        
    # Initial guess: 0 (regular polygon)
    x0 = np.zeros(num_params)
    
    best_params = x0
    best_score = -min_vertex_score(x0)
    
    # Differential Evolution
    if time.time() < deadline - 100:
        try:
            res_de = differential_evolution(
                min_vertex_score,
                bounds,
                maxiter=300,
                popsize=15,
                tol=1e-8,
                update_interval=1,
                mutation=(0.5, 1.5),
                recombination=0.9,
                seed=42
            )
            if res_de.fun < best_score:
                best_score = -res_de.fun
                best_params = res_de.x
        except Exception:
            pass

    # Local Refinement
    rng = np.random.default_rng(42)
    
    restarts = [best_params]
    for i in range(10):
        if time.time() > deadline - 20:
            break
        x_pert = best_params + rng.normal(0, 0.05, size=num_params)
        restarts.append(x_pert)
        
    for x_init in restarts:
        if time.time() > deadline - 10:
            break
        try:
            res_nm = minimize(
                min_vertex_score,
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nm.fun < best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
            
    return fourier_to_points(best_params)
