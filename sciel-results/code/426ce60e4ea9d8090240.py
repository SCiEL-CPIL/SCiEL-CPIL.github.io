"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def halton_sequence(n, base):
    """Generate the first n values of the Halton sequence in base 'base'."""
    result = np.zeros(n)
    f = 1.0
    for i in range(n):
        idx = i + 1
        while idx > 0:
            result[i] += f * (idx % base)
            idx //= base
            f /= base
    return result

def construct():
    # 1. Generate Halton sequence in 3 dimensions
    # Bases should be prime numbers: 2, 3, 5
    bases = [2, 3, 5]
    points = np.column_stack([halton_sequence(N, b) for b in bases])
    
    # Scale to a reasonable range, e.g., [0, 1]^3
    # Halton sequence is already in [0, 1]
    
    # 2. Define the objective function
    def objective(x):
        # Reshape x to (N, D)
        pts = x.reshape(N, D)
        
        # Calculate all pairwise distances
        dists = pdist(pts)
        
        # If any distance is 0, return a large penalty
        if np.any(dists < 1e-10):
            return 1e10
            
        min_dist = np.min(dists)
        max_dist = np.max(dists)
        
        # Ratio squared
        ratio = (max_dist / min_dist) ** 2
        return ratio

    # 3. Optimize
    # Initial guess is the Halton points
    x0 = points.flatten()
    
    # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness
    # well. We set maxiter high to ensure convergence.
    # The problem is small (42 variables), so this should be fast.
    
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Extract optimized points
    opt_points = res.x.reshape(N, D)
    
    return opt_points
# EVOLVE-BLOCK-END
