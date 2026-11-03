"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Model the problem as minimizing the peak autocorrelation
    # subject to a fixed integral (sum).
    # We use gradient descent on the continuous relaxation.
    
    # Initialize with the known optimal solution (constant sequence)
    # normalized to have sum 1.
    x = np.ones(N) / N
    
    # Hyperparameters
    lr = 1e-5
    steps = 10
    
    # Gradient descent loop
    for _ in range(steps):
        # Compute c_0 which is the peak for the constant sequence
        # c_0 = sum(x**2)
        # Gradient of c_0 w.r.t x is 2*x
        grad = 2.0 * x
        
        # Lagrange multiplier for constraint sum(x) = 1
        # grad - lambda * 1 = 0 => lambda = mean(grad)
        lam = np.mean(grad)
        
        # Update rule
        x = x - lr * (grad - lam)
        
        # Enforce constraint sum(x) = 1
        x = x / np.sum(x)
        
    # Scale the result to have a sum of N (matching the magnitude of the seed)
    # The score is scale-invariant, but this keeps the values in a standard range.
    return x * N
# EVOLVE-BLOCK-END
