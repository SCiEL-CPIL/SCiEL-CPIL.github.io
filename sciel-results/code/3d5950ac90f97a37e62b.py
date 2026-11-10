import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    # 1. Generate initial GRS sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:24], dtype=float)
    
    # 2. Define evaluation function
    # We will use a dense grid for initial search and refine peaks
    M = 20000  # Number of points on the unit circle
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    
    # Precompute powers of z for efficient evaluation? 
    # Actually, np.polyval is fine, or we can do matrix mult.
    # Let's use a function that computes P(z) for all z in the grid.
    # P(z) = sum c_k z^k
    # We can create a Vandermonde matrix V where V[j, k] = z_j^k
    # Then P_values = V @ coeffs
    
    # Building Vandermonde matrix once
    # V shape (M, 24)
    # V[j, k] = z[j]**k
    # This is O(M*N) = 20000 * 24 = 480,000 ops. Very fast.
    V = np.vander(z, N=24, increasing=True)
    
    def get_max_mag(c):
        # c is array of 24 coefficients
        # Compute P(z) for all z
        p_vals = V @ c
        mags = np.abs(p_vals)
        
        # Find global max
        idx_max = np.argmax(mags)
        max_val = mags[idx_max]
        
        # Refine around idx_max
        # The peak might be between grid points.
        # We define a local function to minimize -|P(e^{i\theta})|
        # around theta[idx_max]
        
        # Determine neighborhood size
        # The spacing is 2*pi/M
        h = 2 * np.pi / M
        
        # Define objective function for minimization
        # We want to maximize |P(z)|, so minimize -|P(z)|
        def objective(t):
            # t is the angle offset from theta[idx_max]
            z_loc = np.exp(1j * (theta[idx_max] + t))
            # Compute P(z_loc)
            # P(z) = sum c_k z^k
            # We can compute this directly for a single point
            # z_powers = z_loc ** np.arange(24)
            # return -np.abs(np.dot(c, z_powers))
            
            # Faster: use precomputed z powers? No, t changes z.
            # Just compute directly. 24 ops.
            powers = 1.0 + 0j
            sum_val = 0.0 + 0.0j
            for k in range(24):
                sum_val += c[k] * powers
                powers *= z_loc
            return -np.abs(sum_val)
        
        # Use bounded minimization in [-h, h]
        # We need to handle wrap-around, but since h is small, [-h, h] is safe
        # unless idx_max is 0 or M-1.
        # If idx_max is 0, the neighborhood crosses 0.
        # Since the function is periodic and smooth, and we are looking for the local max,
        # minimizing in [-h, h] around the peak is generally sufficient.
        
        res = minimize_scalar(objective, bounds=(-h, h), method='bounded', options={'xatol': 1e-10})
        
        if res.success:
            refined_max = -res.fun
            # It's possible the refined max is slightly higher due to floating point or
            # the peak being exactly on the grid.
            # Also, there might be other peaks. For N=24, the global max is usually unique or few.
            # To be safe, we can check the refined max against the grid max.
            # But the refinement should find the true local max.
            # Is it possible another peak is higher?
            # The grid max is a lower bound for the true max.
            # The refined max is an upper bound for the local max.
            # If the global max is unique, this is fine.
            # For robustness, let's check the top 3 grid peaks and refine them.
            
            # Actually, for N=24, the peaks are well-separated.
            # Let's just refine the top 1. If it fails, we can add more.
            # But let's be safe and refine the top 3 distinct peaks.
            
            # Get top 3 indices
            # Sort indices by magnitude
            # This might be overkill. Let's stick to the top 1 for speed, 
            # as the difference between 1st and 2nd peak is usually significant.
            
            if refined_max > max_val:
                max_val = refined_max
                
        return max_val

    current_max = get_max_mag(coeffs)
    
    start_time = time.time()
    max_time = 115.0  # Leave 5s buffer
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for k in range(24):
            # Flip coefficient k
            old_val = coeffs[k]
            coeffs[k] = -old_val
            
            new_max = get_max_mag(coeffs)
            
            if new_max < current_max - 1e-12:
                current_max = new_max
                improved = True
            else:
                # Revert
                coeffs[k] = old_val
                
    return [int(c) for c in coeffs]
