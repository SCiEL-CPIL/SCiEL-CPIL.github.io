"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a piecewise-linear (hat function) basis.
    This local basis is well-conditioned for BFGS, unlike global polynomials.
    
    We define 10 basis functions:
    - 2 boundary hats (linear ramps)
    - 8 interior hats (triangular peaks)
    
    The basis is defined on the domain [-3, 3].
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.flatten()
    n_points = x_flat.size
    
    # Define knot locations for 10 basis functions
    # Domain [-3, 3], length 6.
    # We want 10 functions.
    # Let's use a grid of 11 points to define 10 intervals, or 10 centers?
    # Standard linear splines with N parameters usually imply N basis functions.
    # Let's place 10 "centers" or "knots" evenly spaced.
    # If we use hat functions centered at x_i, we need to define their support.
    # A simpler approach: define a grid of 11 points from -3 to 3.
    # This creates 10 intervals.
    # Basis function i is 1 at grid point i, 0 at others.
    # This is the standard "hat" basis for linear interpolation.
    
    # Grid points: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3.0
    # Number of points: 11.
    # Number of basis functions: 11? No, params has length 10.
    # So we need 10 basis functions.
    # Let's use 10 knots.
    # If we have 10 knots, we can define 10 hat functions.
    # Let's space 10 knots evenly in [-3, 3].
    # x_knots = np.linspace(-3, 3, 10)
    # Spacing h = 6/9 = 0.666...
    # Hat function i is non-zero in [x_{i-1}, x_{i+1}].
    # For i=0, support [x_0, x_1] (linear ramp up from 0 to 1? Or 1 to 0?)
    # Standard Lagrange-like linear basis:
    # phi_0(x) = (x - x_1)/(x_0 - x_1) for x in [x_0, x_1], 0 otherwise. (Ramp down)
    # phi_i(x) = (x - x_{i-1})/(x_i - x_{i-1}) for x in [x_{i-1}, x_i]
    #             (x_{i+1} - x)/(x_{i+1} - x_i) for x in [x_i, x_{i+1}]
    #             0 otherwise
    # phi_9(x) = (x - x_8)/(x_9 - x_8) for x in [x_8, x_9], 0 otherwise. (Ramp up)
    
    # This ensures partition of unity?
    # Sum of phi_i(x) = 1 for x in [x_0, x_9].
    # This is a good property.
    
    knots = np.linspace(-3.0, 3.0, 10)
    n_basis = 10
    
    # Precompute basis matrix for efficiency
    # T shape: (n_points, 10)
    T = np.zeros((n_points, n_basis), dtype=np.float64)
    
    # Vectorized calculation of basis values
    # For each basis function i, calculate value for all x
    for i in range(n_basis):
        if i == 0:
            # Left boundary: linear ramp from 0 at x_1 to 1 at x_0?
            # Wait, usually we want phi_i(x_i)=1.
            # So phi_0(x_0)=1, phi_0(x_1)=0.
            # Range [x_0, x_1].
            # phi_0(x) = (x_1 - x) / (x_1 - x_0)
            # x < x_0: 0
            # x > x_1: 0
            mask = (x_flat >= knots[0]) & (x_flat <= knots[1])
            T[mask, 0] = (knots[1] - x_flat[mask]) / (knots[1] - knots[0])
            
        elif i == n_basis - 1:
            # Right boundary: linear ramp from 0 at x_8 to 1 at x_9.
            # phi_9(x) = (x - x_8) / (x_9 - x_8)
            # x < x_8: 0
            # x > x_9: 0
            mask = (x_flat >= knots[8]) & (x_flat <= knots[9])
            T[mask, 9] = (x_flat[mask] - knots[8]) / (knots[9] - knots[8])
            
        else:
            # Interior: triangular hat
            # Left slope: [x_{i-1}, x_i]
            # phi_i(x) = (x - x_{i-1}) / (x_i - x_{i-1})
            mask_left = (x_flat >= knots[i-1]) & (x_flat <= knots[i])
            T[mask_left, i] = (x_flat[mask_left] - knots[i-1]) / (knots[i] - knots[i-1])
            
            # Right slope: [x_i, x_{i+1}]
            # phi_i(x) = (x_{i+1} - x) / (x_{i+1} - x_i)
            mask_right = (x_flat >= knots[i]) & (x_flat <= knots[i+1])
            T[mask_right, i] = (knots[i+1] - x_flat[mask_right]) / (knots[i+1] - knots[i])
            
            # Note: At x_i, both masks might apply if boundaries are inclusive.
            # Left: (x_i - x_{i-1})/(x_i - x_{i-1}) = 1
            # Right: (x_{i+1} - x_i)/(x_{i+1} - x_i) = 1
            # So T[x_i, i] = 1. Correct.
            # But wait, if mask_left and mask_right overlap at x_i, we assign 1 twice?
            # No, numpy assignment overwrites. So it stays 1.
            # However, if we use `+=` it would be 2. We use `=`, so it's fine.
            # But we must ensure no other basis function is non-zero at x_i.
            # phi_{i-1} is non-zero in [x_{i-2}, x_i]. At x_i, phi_{i-1} = 0.
            # phi_{i+1} is non-zero in [x_i, x_{i+2}]. At x_i, phi_{i+1} = 0.
            # So only phi_i is non-zero at x_i. Correct.
            
            # Check time
            if time.time() > deadline:
                # Return current best effort (zeros if not started, partial if started)
                # Since this is fast, this is unlikely to trigger.
                return T @ params

    # Handle x outside [-3, 3]?
    # The problem states x is drawn from [-3, 3].
    # So no extrapolation needed.
    # If x < -3 or x > 3, T is all zeros, prediction 0.
    # This is acceptable given the domain constraint.
    
    prediction = T @ params
    
    return prediction.reshape(x.shape)
# EVOLVE-BLOCK-END
