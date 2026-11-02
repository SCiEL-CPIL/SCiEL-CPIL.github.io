import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 14

def _min_triangle_area_from_gaps(gaps):
    """
    Calculate the minimum triangle area for points on a unit circle
    defined by angular gaps.
    gaps: array of length N, summing to 2*pi.
    """
    n = len(gaps)
    # Cumulative angles
    angles = np.zeros(n)
    angles[1:] = np.cumsum(gaps[:-1])
    
    min_area = float('inf')
    # Iterate over all triples
    # We can optimize this, but N=14 is small (364 triples)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # The angular gaps between i, j, k in circular order
                # Let the gaps be alpha (i to j), beta (j to k), gamma (k to i)
                # alpha = angles[j] - angles[i]
                # beta = angles[k] - angles[j]
                # gamma = 2*pi - (angles[k] - angles[i])
                
                # Calculate areas using the formula 0.5 * (sin(alpha) + sin(beta) + sin(gamma))
                # Note: This formula gives the area of the triangle inscribed in the unit circle.
                
                alpha = angles[j] - angles[i]
                beta = angles[k] - angles[j]
                gamma = 2 * np.pi - (angles[k] - angles[i])
                
                area = 0.5 * (np.sin(alpha) + np.sin(beta) + np.sin(gamma))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area_from_gaps(gaps):
    """
    Calculate the area of the convex hull (polygon) for points on a unit circle
    defined by angular gaps.
    """
    # Area = 0.5 * sum(sin(gaps))
    return 0.5 * np.sum(np.sin(gaps))

def _score_from_gaps(gaps):
    """
    Calculate the score: min triangle area / hull area.
    """
    min_area = _min_triangle_area_from_gaps(gaps)
    hull_area = _hull_area_from_gaps(gaps)
    if hull_area < 1e-9:
        return -1e9
    return min_area / hull_area

def _objective(gaps_params):
    """
    Objective function for the optimizer.
    gaps_params: 13 values. The 14th gap is derived.
    """
    # Ensure gaps are positive
    if np.any(gaps_params <= 0):
        return 1e9
        
    sum_params = np.sum(gaps_params)
    if sum_params >= 2 * np.pi:
        return 1e9
        
    gaps = np.zeros(14)
    gaps[:13] = gaps_params
    gaps[13] = 2 * np.pi - sum_params
    
    if gaps[13] <= 0:
        return 1e9
        
    score = _score_from_gaps(gaps)
    return -score # Minimize negative score

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initial guess: Regular polygon
    regular_gap = 2 * np.pi / N
    initial_params = np.full(13, regular_gap)
    
    best_params = initial_params
    best_score = _score_from_gaps(np.append(best_params, 2*np.pi - np.sum(best_params)))
    
    # Try Nelder-Mead locally
    try:
        result = minimize(_objective, initial_params, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if time.time() < deadline:
            candidate_params = result.x
            # Validate
            sum_c = np.sum(candidate_params)
            if np.all(candidate_params > 0) and sum_c < 2 * np.pi:
                gaps_c = np.zeros(14)
                gaps_c[:13] = candidate_params
                gaps_c[13] = 2 * np.pi - sum_c
                score_c = _score_from_gaps(gaps_c)
                if score_c > best_score:
                    best_score = score_c
                    best_params = candidate_params
    except Exception:
        pass
    
    # If time remains, try Differential Evolution with small population
    if time.time() < deadline - 30:
        try:
            bounds = [(1e-6, 2*np.pi - 1e-6)] * 13
            # Use a small population to keep it fast
            de_result = differential_evolution(_objective, bounds, maxiter=50, 
                                               popsize=10, tol=1e-4, 
                                               seed=42, polish=False)
            
            if time.time() < deadline:
                candidate_params = de_result.x
                sum_c = np.sum(candidate_params)
                if np.all(candidate_params > 0) and sum_c < 2 * np.pi:
                    gaps_c = np.zeros(14)
                    gaps_c[:13] = candidate_params
                    gaps_c[13] = 2 * np.pi - sum_c
                    score_c = _score_from_gaps(gaps_c)
                    if score_c > best_score:
                        best_score = score_c
                        best_params = candidate_params
        except Exception:
            pass

    # Construct final points
    gaps = np.zeros(14)
    gaps[:13] = best_params
    gaps[13] = 2 * np.pi - np.sum(best_params)
    
    # Ensure gaps are positive and sum to 2pi
    if np.any(gaps <= 0) or abs(np.sum(gaps) - 2*np.pi) > 1e-9:
        # Fallback to regular polygon
        gaps = np.full(14, 2*np.pi/14)

    angles = np.zeros(14)
    angles[1:] = np.cumsum(gaps[:-1])
    
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return points

# EVOLVE-BLOCK-END
