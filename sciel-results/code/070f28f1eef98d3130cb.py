import numpy as np
from scipy.optimize import minimize
import time

N = 13

def compute_score(points):
    """
    Compute the ratio of the minimum triangle area to the convex hull area.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    # Since points are on the unit circle and ordered, the hull is the polygon itself.
    # However, to be safe and general, we use the shoelace formula on the ordered points.
    # The points are generated from sorted angles, so they are in CCW order.
    x = points[:, 0]
    y = points[:, 1]
    hull_area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    
    if hull_area < 1e-10:
        return 0.0

    # Compute minimum triangle area
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area of triangle with vertices i, j, k
                # Using determinant formula
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def objective(gaps):
    """
    Objective function to minimize (negative of score).
    gaps: array of 12 angular gaps between consecutive points.
    The 13th gap is 2*pi - sum(gaps).
    """
    total_gap = 2 * np.pi
    sum_gaps = np.sum(gaps)
    # Ensure all gaps are positive and sum is correct
    # We use a barrier or just constrain via bounds in optimizer.
    # For Nelder-Mead, we need to handle invalid states.
    if np.any(gaps <= 1e-6):
        return 1.0  # Large penalty (since we minimize negative score, positive is bad)
    if sum_gaps > total_gap + 1e-6:
        return 1.0
        
    last_gap = total_gap - sum_gaps
    if last_gap <= 1e-6:
        return 1.0

    all_gaps = np.append(gaps, last_gap)
    
    # Construct angles
    angles = np.cumsum(np.concatenate(([0], all_gaps)))
    # angles[0] is 0, angles[-1] is 2*pi (which is same as 0)
    # We need N points, so we take the first N angles
    # The points are at angles[0], angles[1], ..., angles[N-1]
    # Note: angles has length N+1 because of the cumulative sum including the last gap back to 2*pi.
    # We only need the first N points.
    pts = np.column_stack((np.cos(angles[:N]), np.sin(angles[:N])))
    
    score = compute_score(pts)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Initial guess: regular polygon
    regular_gaps = np.full(N - 1, 2 * np.pi / N)
    
    # Bounds for gaps: must be positive and sum to 2*pi
    # We can't easily set sum constraint in Nelder-Mead, but we can set individual bounds.
    # Min gap: very small. Max gap: 2*pi - (N-1)*min_gap.
    min_gap = 1e-4
    max_gap = 2 * np.pi - (N - 1) * min_gap
    
    bounds = [(min_gap, max_gap) for _ in range(N - 1)]
    
    # Try to optimize
    try:
        # Nelder-Mead is good for non-smooth functions
        res = minimize(
            objective, 
            regular_gaps, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        # Check time
        if time.time() > deadline:
            # Return regular polygon if we ran out of time
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.column_stack((np.cos(angles), np.sin(angles)))
            
        best_gaps = res.x
        last_gap = 2 * np.pi - np.sum(best_gaps)
        all_gaps = np.append(best_gaps, last_gap)
        angles = np.cumsum(np.concatenate(([0], all_gaps)))
        pts = np.column_stack((np.cos(angles[:N]), np.sin(angles[:N])))
        
        # Verify score
        final_score = compute_score(pts)
        
        # Compare with regular polygon score
        reg_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg_pts = np.column_stack((np.cos(reg_angles), np.sin(reg_angles)))
        reg_score = compute_score(reg_pts)
        
        if final_score >= reg_score:
            return pts
        else:
            return reg_pts
            
    except Exception as e:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We place it in the block as requested by the prompt structure if needed, 
# but the prompt says "Keep the function names and signatures... Output the COMPLETE program".
# The current program has EVOLVE-BLOCK markers. I will provide the full script.
# EVOLVE-BLOCK-END
