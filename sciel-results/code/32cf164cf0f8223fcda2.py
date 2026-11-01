import numpy as np
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Objective function to maximize: min triangle area / hull area
    # We optimize over the angular gaps between consecutive points on a circle.
    # This guarantees convexity and points on the boundary.
    
    def score_from_gaps(gaps):
        # gaps are the angular differences between consecutive points
        # sum(gaps) should be 2*pi, but optimizer might drift, so we normalize or constrain
        # Let's assume gaps are positive and sum to 2*pi.
        
        # Calculate angles
        angles = np.cumsum(np.concatenate(([0.0], gaps)))
        angles = angles % (2 * np.pi)
        
        # Calculate points
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Calculate Hull Area (Area of polygon defined by pts in order)
        # Shoelace formula
        x = pts[:, 0]
        y = pts[:, 1]
        area_hull = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        
        if area_hull < 1e-9:
            return 0.0
            
        # Calculate all triangle areas
        # There are 364 triangles for N=14
        min_area = float('inf')
        # Vectorized calculation is faster
        # i, j, k indices
        # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
        
        # To save time, we can just compute the min.
        # 14 choose 3 is small.
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p_i = pts[i]
                    p_j = pts[j]
                    p_k = pts[k]
                    # Cross product magnitude / 2
                    area = 0.5 * abs((p_j[0] - p_i[0]) * (p_k[1] - p_i[1]) - (p_j[1] - p_i[1]) * (p_k[0] - p_i[0]))
                    if area < min_area:
                        min_area = area
                        if min_area < 1e-9:
                            return 0.0
        
        return min_area / area_hull

    def neg_score_from_gaps(gaps):
        # We want to maximize score, so minimize negative score
        # Ensure gaps are positive. If any gap <= 0, return large penalty
        if np.any(gaps <= 1e-6):
            return 1e6
        # Normalize gaps to sum to 2*pi to maintain the circle constraint strictly
        # Actually, it's better to let the optimizer work on unconstrained variables
        # and map them to positive gaps summing to 2*pi.
        # Let x be unconstrained. gaps_i = exp(x_i) / sum(exp(x)) * 2*pi
        # This ensures positivity and sum constraint.
        exp_gaps = np.exp(gaps)
        total = np.sum(exp_gaps)
        normalized_gaps = (exp_gaps / total) * (2 * np.pi)
        
        return -score_from_gaps(normalized_gaps)

    # Initial guess: Regular polygon
    # Equal gaps: 2*pi / 14
    # In log-space: log(1/14) for all, but we just need the shape.
    # If gaps are equal, exp(x) are equal. So x_i = 0 for all i is a good start.
    x0 = np.zeros(N)
    
    # Optimization
    try:
        from scipy.optimize import minimize
        
        # Use Nelder-Mead or Powell for robustness without gradients
        # Powell is good for this smooth, symmetric problem
        res = minimize(neg_score_from_gaps, x0, method='Powell', 
                       options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
        
        # Check time
        if time.time() > deadline:
             # Fallback to regular polygon if optimization took too long
             pass
        else:
             # Extract best gaps from result
             if res.success or res.fun < 0: # If it improved or succeeded
                 best_gaps_log = res.x
                 exp_gaps = np.exp(best_gaps_log)
                 total = np.sum(exp_gaps)
                 best_gaps = (exp_gaps / total) * (2 * np.pi)
                 
                 # Generate final points
                 angles = np.cumsum(np.concatenate(([0.0], best_gaps)))
                 angles = angles % (2 * np.pi)
                 points = np.column_stack((np.cos(angles), np.sin(angles)))
                 return points
             
    except Exception:
        pass
    
    # Fallback: Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return points

