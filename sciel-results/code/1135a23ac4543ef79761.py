import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13
MAX_TIME = 145.0  # Safety margin before 150s limit

def compute_scores(points):
    """
    Compute the minimum triangle area and the convex hull area.
    Returns (min_area, hull_area).
    """
    n = points.shape[0]
    # Compute all triangle areas vectorized
    # i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    ii = i_idx[mask]
    jj = j_idx[mask]
    kk = k_idx[mask]
    
    v1 = points[jj] - points[ii]
    v2 = points[kk] - points[ii]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    min_area = np.min(areas)
    
    # Compute hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        hull_area = 1e-10
        
    return min_area, hull_area

def score(points):
    """
    Score = min_triangle_area / hull_area
    """
    min_area, hull_area = compute_scores(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def coordinate_descent_polish(points, max_iters=50, tol=1e-8):
    """
    Refine points using coordinate descent.
    For each point and each coordinate, perform a local line search.
    """
    best_points = points.copy()
    best_score = score(best_points)
    
    # Use a simple bisection or golden section search for 1D optimization
    # Since we want to maximize score, and the landscape is somewhat noisy but locally smooth,
    # we can try small steps.
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            for d in range(2):
                # Current value
                val = best_points[i, d]
                current_score = best_score
                
                # Try expanding a window around val
                # The optimal value is likely close to the current one.
                # We will search in a small interval [val - delta, val + delta]
                # Start with a small delta and increase if we hit the boundary.
                delta = 0.01
                found_peak = False
                
                # We perform a few iterations of interval refinement
                for _iter in range(10):
                    low = val - delta
                    high = val + delta
                    
                    # Evaluate at low, mid, high
                    p_low = best_points.copy()
                    p_low[i, d] = low
                    s_low = score(p_low)
                    
                    p_high = best_points.copy()
                    p_high[i, d] = high
                    s_high = score(p_high)
                    
                    p_mid = best_points.copy()
                    p_mid[i, d] = val
                    s_mid = current_score
                    
                    # If the peak is at the boundary, expand
                    if s_low > s_mid and s_low > s_high:
                        # Peak is likely to the left
                        val = low
                        delta *= 0.5
                        # Update best if better
                        if s_low > best_score:
                            best_score = s_low
                            best_points = p_low.copy()
                            improved = True
                        continue
                    elif s_high > s_mid and s_high > s_low:
                        # Peak is likely to the right
                        val = high
                        delta *= 0.5
                        if s_high > best_score:
                            best_score = s_high
                            best_points = p_high.copy()
                            improved = True
                        continue
                    else:
                        # Peak is in between, narrow down
                        # Use golden section or simple trisection
                        m1 = low + (high - low) / 3.0
                        m2 = high - (high - low) / 3.0
                        
                        p_m1 = best_points.copy()
                        p_m1[i, d] = m1
                        s_m1 = score(p_m1)
                        
                        p_m2 = best_points.copy()
                        p_m2[i, d] = m2
                        s_m2 = score(p_m2)
                        
                        if s_m1 > s_m2:
                            # Peak is in [low, m2]
                            high = m2
                            val = m1 # Move center towards better side? No, just narrow interval
                            # Actually, let's just narrow the interval for the next iteration
                            # We keep 'val' as the current best guess for next delta calc
                            # But for the next loop, we use the narrowed [low, high]
                            if s_m1 > best_score:
                                best_score = s_m1
                                best_points = p_m1.copy()
                                improved = True
                        else:
                            # Peak is in [m1, high]
                            low = m1
                            if s_m2 > best_score:
                                best_score = s_m2
                                best_points = p_m2.copy()
                                improved = True
                        
                        # Update val to the center of the new interval for next delta calc?
                        # Or just let the loop handle the new low/high.
                        # The 'val' variable is used to set low/high in the next iteration.
                        # So we should update val to the best point found so far in this coordinate.
                        # If s_m1 > s_m2, best is m1. Else m2.
                        if s_m1 >= s_m2:
                            val = m1
                        else:
                            val = m2
                            
        if not improved:
            break
            
    return best_points, best_score

def simulated_annealing(points, time_limit):
    """
    Simulated Annealing to find a good global configuration.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    # Number of iterations per second estimate
    # We will run until time_limit
    while time.time() < start_time + time_limit:
        # Generate a new candidate by perturbing one point
        idx = np.random.randint(0, N)
        new_points = current_points.copy()
        
        # Gaussian perturbation
        # Scale perturbation with temperature
        step_size = T * 0.1
        new_points[idx] += np.random.normal(0, step_size, 2)
        
        new_score = score(new_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # 1. Initialize with a regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area? The score is ratio, so scaling doesn't matter.
    # But let's keep them in a reasonable range, e.g., unit circle.
    
    best_points = init_points.copy()
    best_score = score(best_points)
    
    # 2. Run Simulated Annealing for a significant portion of the time
    # Let's give SA 80% of the time, and CD 20%
    sa_time = MAX_TIME * 0.8
    sa_points, sa_score = simulated_annealing(init_points, sa_time)
    
    if sa_score > best_score:
        best_score = sa_score
        best_points = sa_points
    
    # 3. Coordinate Descent Polish
    # Run CD from the best point found
    if time.time() < deadline - 2:
        cd_points, cd_score = coordinate_descent_polish(best_points, max_iters=20)
        if cd_score > best_score:
            best_score = cd_score
            best_points = cd_points
            
            # Run CD again if time permits
            if time.time() < deadline - 2:
                cd_points2, cd_score2 = coordinate_descent_polish(best_points, max_iters=10)
                if cd_score2 > best_score:
                    best_score = cd_score2
                    best_points = cd_points2

    return best_points
