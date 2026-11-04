import numpy as np
from scipy.optimize import differential_evolution

N = 14

def construct():
    # We optimize the N-1 free angular gaps. The N-th gap is determined by the sum constraint.
    # Bounds: Each gap must be positive.
    # A reasonable lower bound is very small, upper bound is 2*pi.
    
    def objective(gaps):
        # gaps: array of length N-1
        # Calculate the last gap
        sum_gaps = np.sum(gaps)
        last_gap = 2 * np.pi - sum_gaps
        
        # Check validity
        if last_gap <= 1e-9 or np.any(gaps <= 1e-9):
            return 0.0 # Return 0 (minimize negative score, so 0 is bad)
            
        all_gaps = np.concatenate([gaps, [last_gap]])
        
        # Calculate cumulative angles
        angles = np.zeros(N)
        angles[1:] = np.cumsum(all_gaps[:N-1])
        # angles are in [0, 2*pi)
        
        min_area = float('inf')
        
        # Iterate over all triples
        # To be distinct from the current program's loop structure, 
        # I will precompute sine differences or use a different iteration order.
        
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    # Arcs
                    a = angles[j] - angles[i]
                    b = angles[k] - angles[j]
                    c = 2 * np.pi - a - b
                    
                    # Area is proportional to sin(a) + sin(b) + sin(c)
                    # Note: sin(c) = sin(2pi - a - b) = -sin(a+b)
                    # Area = 0.5 * (sin(a) + sin(b) - sin(a+b))
                    area_val = 0.5 * (np.sin(a) + np.sin(b) - np.sin(a + b))
                    
                    if area_val < min_area:
                        min_area = area_val
                        
        # We want to maximize min_area.
        # differential_evolution minimizes.
        # So we return -min_area.
        # If min_area is 0 (degenerate), return 0.
        if min_area <= 1e-9:
            return 0.0
            
        return -min_area

    # Bounds for the first N-1 gaps
    # They must sum to less than 2*pi.
    # Simple bounds: (1e-6, 2*pi - 1e-6)
    bounds = [(1e-6, 2 * np.pi - 1e-6)] * (N - 1)
    
    # Initial guess: Uniform
    x0 = np.full(N - 1, 2 * np.pi / N)
    
    # Run Differential Evolution
    # seed=42 for reproducibility
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=1000, 
        tol=1e-8,
        polish=True # Use L-BFGS-B for final polishing
    )
    
    if result.success:
        opt_gaps = result.x
    else:
        # Fallback to uniform
        opt_gaps = np.full(N - 1, 2 * np.pi / N)
        
    # Construct the full gap array
    full_gaps = np.concatenate([opt_gaps, [2 * np.pi - np.sum(opt_gaps)]])
    
    # Calculate angles
    angles = np.zeros(N)
    angles[1:] = np.cumsum(full_gaps[:N-1])
    
    # Points on unit circle
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
