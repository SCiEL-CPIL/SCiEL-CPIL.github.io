"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_score_from_gaps(gaps):
    """
    Compute score given angular gaps between consecutive points.
    gaps: array of length N, where gaps[i] is angle from point i-1 to i.
    Sum of gaps must be 2*pi.
    """
    # Check validity
    if np.any(gaps <= 1e-6):
        return 0.0
    
    # Calculate angles
    angles = np.cumsum(gaps)
    # angles[0] is gap[0], which is angle of point 0 relative to -1 (which is 0).
    # So point 0 is at angle gaps[0].
    # Wait, let's define:
    # Point 0 is at angle 0.
    # Point 1 is at angle g1.
    # Point 2 is at angle g1+g2.
    # ...
    # Point N-1 is at angle sum(g1..g_{N-1}).
    # Gap g_N is the gap from N-1 to 0 (wrapping around).
    # So we optimize g1..g_{N-1}.
    
    # In this function, we assume gaps is the full array of N gaps.
    # Let's just take the first N-1 as variables and compute the last.
    
    pass # This function is a draft, let's integrate into objective


def _objective(params):
    """
    Objective function.
    params: (N-1,) array of positive gaps g1..g_{N-1}.
    """
    g = params
    # Ensure positivity
    if np.any(g <= 1e-6):
        return 1e6 # Large penalty
    
    g_n = 2 * np.pi - np.sum(g)
    if g_n <= 1e-6:
        return 1e6 # Large penalty
    
    # Construct full gaps array
    full_gaps = np.concatenate([g, [g_n]])
    
    # Construct angles
    # Point 0 at 0
    # Point i at sum(full_gaps[1..i])? 
    # Let's say full_gaps[i] is the gap from point i-1 to point i.
    # So angle[0] = 0.
    # angle[1] = full_gaps[1]
    # angle[2] = full_gaps[1] + full_gaps[2]
    # ...
    # angle[i] = sum(full_gaps[1..i])
    
    angles = np.zeros(N)
    angles[1:] = np.cumsum(full_gaps[1:])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Calculate Hull Area
    # Points are already in angular order 0 to 2pi.
    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area_sum = np.sum(x * y_next - x_next * y)
    hull_area = 0.5 * abs(area_sum)
    
    if hull_area < 1e-12:
        return 1e6
        
    # Find minimum triangle area
    min_area = float('inf')
    n = N
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    score = min_area / hull_area
    return -score


def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    
    Returns: (N, 2) array of points
    """
    # Initial guess: equal gaps
    initial_gap = 2 * np.pi / N
    initial_params = np.full(N - 1, initial_gap)
    
    # Use Nelder-Mead to optimize the gaps
    result = minimize(
        _objective,
        initial_params,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    optimal_gaps = result.x
    g_n = 2 * np.pi - np.sum(optimal_gaps)
    full_gaps = np.concatenate([optimal_gaps, [g_n]])
    
    # Construct angles
    angles = np.zeros(N)
    angles[1:] = np.cumsum(full_gaps[1:])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
# EVOLVE-BLOCK-END
