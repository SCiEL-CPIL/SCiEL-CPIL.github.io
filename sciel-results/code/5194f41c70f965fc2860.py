import numpy as np
from scipy.optimize import minimize
import time

N = 14

def construct():
    # Start time to manage budget
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initial guess: uniform distribution (regular N-gon)
    delta_init = np.full(N-1, 2 * np.pi / N)
    
    def objective(x):
        # x: array of first N-1 gaps
        # Check for invalid gaps (must be positive)
        if np.any(x <= 0):
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        # Check if the last gap is valid
        if deltas[N-1] <= 0:
            return 0.0
            
        # Calculate cumulative angles (starting at 0)
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Vectorized calculation of minimum triangle area numerator
        # We need to find min over all i<j<k of sin(phi[j]-phi[i]) + sin(phi[k]-phi[j]) + sin(2pi - (phi[k]-phi[i]))
        # Let a = phi[j]-phi[i], b = phi[k]-phi[j], c = 2pi - a - b.
        # Note: c is the arc from k to i going the other way? No, the triangle is formed by chords.
        # The area of triangle with vertices at angles a1, a2, a3 is 0.5 * R^2 * (sin(a2-a1) + sin(a3-a2) + sin(a1-a3+2pi))
        # Since we are on a circle, the "central angles" subtended by the sides are not the same as the triangle's interior angles.
        # The formula Area = 0.5 * R^2 * (sin(theta1) + sin(theta2) + sin(theta3)) where theta_i are the arcs between consecutive points on the circle in the order of the triangle vertices.
        # For points i, j, k in counter-clockwise order, the arcs are:
        # arc_ij = phi[j] - phi[i]
        # arc_jk = phi[k] - phi[j]
        # arc_ki = 2*pi - (phi[k] - phi[i])
        
        # We can compute this for all triples using broadcasting.
        # phi shape: (N,)
        # We want all pairs (i, j) with i < j.
        # Then for each (i, j), all k > j.
        
        # To speed up, we can just loop i, j and vectorize k? Or loop i and vectorize j, k?
        # N=14 is small, but the Python loop overhead is high.
        
        min_num = float('inf')
        
        # Precompute sin of all pairwise differences?
        # diff[i, j] = phi[j] - phi[i]
        # We need sin(diff[i, j]) + sin(diff[j, k]) + sin(2pi - diff[i, k])
        
        # Let's stick to the loop but optimize the inner part if possible.
        # Actually, for N=14, 14*13*12/6 = 364 triples. This is small.
        # The bottleneck in the original code was likely the optimization iterations, not the objective eval itself, 
        # unless the objective was called millions of times. Nelder-Mead calls it many times.
        # Let's make the objective as fast as possible.
        
        # Vectorized approach for objective:
        # Create indices for all triples
        # This might be overkill for N=14. The loop is ~364 iterations.
        # Let's keep the loop but ensure it's tight.
        
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        # Calculate hull area denominator
        # Hull area for points on circle is sum of 0.5 * sin(delta) for each gap?
        # No, the area of the polygon formed by points on a circle with angular gaps d1...dN is
        # 0.5 * R^2 * sum(sin(di)).
        # R=1.
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_num / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_ratio = -float('inf')
    best_x = None
    
    # Define multiple starting points
    starts = []
    # 1. Regular polygon
    starts.append(np.full(N-1, 2 * np.pi / N))
    
    # 2. Slightly perturbed regular polygon
    rng = np.random.default_rng(42)
    for _ in range(5):
        pert = rng.normal(0, 0.1, N-1)
        starts.append(delta_init + pert)
        
    # 3. Clustered points? (Unlikely to be optimal for max min area, but good for search space coverage)
    # 4. Alternating large/small gaps
    alt = np.zeros(N-1)
    for i in range(N-1):
        if i % 2 == 0:
            alt[i] = 2 * np.pi / N * 1.2
        else:
            alt[i] = 2 * np.pi / N * 0.8
    # Normalize to sum to 2*pi? The last gap adjusts.
    starts.append(alt)
    
    # Run optimization from each start
    for x0 in starts:
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        try:
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                bounds=bounds,
                options={
                    'xatol': 1e-12, 
                    'fatol': 1e-12, 
                    'maxiter': 5000, 
                    'adaptive': True,
                    'maxfev': 20000
                }
            )
            
            if result.success or result.fun < objective(x0): # Check if it improved
                if -result.fun > best_ratio:
                    best_ratio = -result.fun
                    best_x = result.x
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to uniform if optimization fails
        opt_deltas = np.full(N, 2 * np.pi / N)
    else:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = best_x
        opt_deltas[N-1] = 2 * np.pi - np.sum(best_x)
        
    # Ensure validity
    if np.any(opt_deltas <= 0):
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
