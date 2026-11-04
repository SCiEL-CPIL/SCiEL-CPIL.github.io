import numpy as np
from scipy.optimize import minimize

N = 14
TWO_PI = 2.0 * np.pi

def construct():
    # We are optimizing the angular positions of N points on a circle.
    # The objective is to maximize the minimum triangle area divided by the convex hull area.
    # For points on a circle, the triangle area is proportional to sin(a) + sin(b) + sin(c)
    # where a, b, c are the arc lengths between the three points.
    # The convex hull area is proportional to sum(sin(deltas)) where deltas are the gaps between consecutive points.
    # We parameterize the gaps between consecutive points. There are N gaps summing to 2*pi.
    # We use N-1 free variables (the first N-1 gaps) and the last is determined.
    
    # Initial guess: uniform distribution (regular polygon)
    delta_init = np.full(N-1, TWO_PI / N)
    
    def objective(x):
        # x: array of first N-1 gaps
        if np.any(x <= 1e-12):
            return 0.0
        if np.sum(x) >= TWO_PI:
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = TWO_PI - np.sum(x)
        
        if deltas[N-1] <= 1e-12:
            return 0.0
            
        # Calculate cumulative angles
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate the minimum triangle area numerator
        # For each triple (i, j, k), the area is proportional to sin(phi[j]-phi[i]) + sin(phi[k]-phi[j]) + sin(2*pi - (phi[k]-phi[i]))
        # But we need to be careful with the arcs. The arcs are the differences in angles modulo 2*pi.
        # Since phi is sorted and in [0, 2*pi), the arcs are:
        # a = phi[j] - phi[i]
        # b = phi[k] - phi[j]
        # c = 2*pi - (phi[k] - phi[i]) = 2*pi - a - b
        min_num = float('inf')
        
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = TWO_PI - a - b
                    
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        # Calculate hull area denominator
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_num / den
        return -ratio # We minimize the negative ratio to maximize the ratio

    # Bounds for each gap
    # They must be positive and sum to 2*pi
    bounds = [(1e-12, TWO_PI - 1e-12)] * (N-1)
    
    # Use Nelder-Mead as it doesn't require gradients and is robust for small N
    # We start from the regular polygon which is a strong initial guess.
    # We also try a few other initial guesses to ensure we find the global optimum.
    
    best_result = None
    best_score = -float('inf')
    
    # Initial guesses:
    # 1. Regular polygon
    # 2. Slightly perturbed regular polygon
    # 3. A few random perturbations
    
    initial_guesses = [delta_init.copy()]
    
    # Add a few random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbation = rng.uniform(0.9, 1.1, N-1)
        guess = delta_init * perturbation
        # Normalize to sum to 2*pi - small epsilon to keep the last gap positive
        guess = guess * (TWO_PI - 1e-9) / np.sum(guess)
        if np.all(guess > 1e-12):
            initial_guesses.append(guess)
    
    for guess in initial_guesses:
        result = minimize(
            objective, 
            guess, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={
                'xatol': 1e-12, 
                'fatol': 1e-12, 
                'maxiter': 5000, 
                'adaptive': True
            }
        )
        
        if result.success:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_result = result
    
    if best_result is not None:
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = best_result.x
        opt_deltas[N-1] = TWO_PI - np.sum(best_result.x)
    else:
        # Fallback to uniform if optimization fails
        opt_deltas = np.full(N, TWO_PI / N)
        
    # Construct points on the unit circle
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
