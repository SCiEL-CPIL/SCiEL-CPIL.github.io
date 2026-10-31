import numpy as np
from scipy.optimize import differential_evolution
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    # points: (N, 2)
    # returns (N, N) matrix of squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective_phases(phi):
    # phi: array of 16 angles
    # Construct coefficients c_k = exp(1j * phi_k)
    coeffs = np.exp(1j * phi)
    
    # The polynomial is P(z) = sum_{k=0}^{15} c_k z^k
    # numpy.poly1d or np.roots expects coefficients in descending order of power
    # So coeffs[15] is for z^15, coeffs[0] is for z^0
    # np.roots takes [a_n, a_{n-1}, ..., a_0]
    coeff_desc = coeffs[::-1]
    
    try:
        roots = np.roots(coeff_desc)
    except Exception:
        return 1e10
        
    # Map complex roots to 2D points
    points = np.column_stack((roots.real, roots.imag))
    
    # Check for distinct points (min distance not zero)
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals < 1e-12):
        return 1e10
        
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    # The score is (max_dist / min_dist)^2 = (sqrt(d_max_sq) / sqrt(d_min_sq))^2 = d_max_sq / d_min_sq
    return d_max_sq / d_min_sq

def construct():
    # Bounds for each phase angle [0, 2*pi]
    bounds = [(0, 2 * np.pi)] * N
    
    # Use Differential Evolution
    # maxiter and popsize are tuned to balance exploration and time limit
    # 120s limit. DE is stochastic but can be fast with appropriate settings.
    try:
        res = differential_evolution(
            objective_phases, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5,
            seed=42,
            polish=True, # Use L-BFGS-B for final polishing
            init='sobol'
        )
        
        if res.success:
            best_phases = res.x
        else:
            best_phases = res.x # Use best found even if not fully converged
            
        # Convert best phases to points
        coeffs = np.exp(1j * best_phases)
        coeff_desc = coeffs[::-1]
        roots = np.roots(coeff_desc)
        points = np.column_stack((roots.real, roots.imag))
        
        # Ensure distinctness
        d_sq = pairwise_distances_sq(points)
        mask = ~np.eye(N, dtype=bool)
        d_sq_vals = d_sq[mask]
        if np.min(d_sq_vals) > 1e-12:
            return points
    except Exception:
        pass
        
    # Fallback: Return a valid configuration if optimization fails
    # Use a simple grid or random points to ensure a valid return
    # We must return an array of shape (16, 2)
    # Let's use the hexagonal seed from the old code as a fallback to ensure validity
    hex_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 2:
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                hex_candidates.append((x, y))
    hex_arr = np.array(hex_candidates)
    d_orig = np.sum(hex_arr**2, axis=1)
    sorted_indices = np.argsort(d_orig)
    fallback = hex_arr[sorted_indices[:N]]
    fallback -= np.mean(fallback, axis=0)
    return fallback

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective_phases(phi):
    coeffs = np.exp(1j * phi)
    coeff_desc = coeffs[::-1]
    
    try:
        roots = np.roots(coeff_desc)
    except Exception:
        return 1e10
        
    points = np.column_stack((roots.real, roots.imag))
    
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals < 1e-12):
        return 1e10
        
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def construct():
    bounds = [(0, 2 * np.pi)] * N
    
    try:
        res = differential_evolution(
            objective_phases, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-5,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        best_phases = res.x
        coeffs = np.exp(1j * best_phases)
        coeff_desc = coeffs[::-1]
        roots = np.roots(coeff_desc)
        points = np.column_stack((roots.real, roots.imag))
        
        d_sq = pairwise_distances_sq(points)
        mask = ~np.eye(N, dtype=bool)
        d_sq_vals = d_sq[mask]
        if np.min(d_sq_vals) > 1e-12:
            return points
    except Exception:
        pass
        
    hex_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 2:
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                hex_candidates.append((x, y))
    hex_arr = np.array(hex_candidates)
    d_orig = np.sum(hex_arr**2, axis=1)
    sorted_indices = np.argsort(d_orig)
    fallback = hex_arr[sorted_indices[:N]]
    fallback -= np.mean(fallback, axis=0)
    return fallback
# EVOLVE-BLOCK-END
