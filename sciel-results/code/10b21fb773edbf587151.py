"""Symbolic regression: rich basis with better conditioning for BFGS convergence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a diverse basis of 10 functions with
    improved conditioning to help BFGS converge from the all-ones starting point.
    
    params[0]:  1 (constant)
    params[1]:  x
    params[2]:  x^2
    params[3]:  x^3
    params[4]:  sin(x)
    params[5]:  cos(x)
    params[6]:  sin(2x)
    params[7]:  cos(2x)
    params[8]:  1/(1 + x^2)   (Lorentzian-like)
    params[9]:  exp(-x^2/4)   (Gaussian-like)
    
    The evaluator will fit these 10 parameters via BFGS starting from ones.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Build the 10 basis functions
    c0 = np.ones(n, dtype=np.float64)
    c1 = x
    c2 = x * x
    c3 = c2 * x
    s1 = np.sin(x)
    s2 = np.cos(x)
    s3 = np.sin(2.0 * x)
    s4 = np.cos(2.0 * x)
    s5 = 1.0 / (1.0 + c2)
    s6 = np.exp(-c2 / 4.0)
    
    # Stack into (n, 10) matrix
    V = np.empty((n, 10), dtype=np.float64)
    V[:, 0] = c0
    V[:, 1] = c1
    V[:, 2] = c2
    V[:, 3] = c3
    V[:, 4] = s1
    V[:, 5] = s2
    V[:, 6] = s3
    V[:, 7] = s4
    V[:, 8] = s5
    V[:, 9] = s6
    
    # Apply a well-conditioned scaling to the parameter space.
    # We scale the columns of V so that each basis function has approximately
    # unit RMS magnitude over the domain [-3, 3]. This makes the Hessian of the
    # MSE loss better conditioned, helping BFGS converge more reliably from
    # the all-ones initialization.
    
    # Compute RMS of each column over a grid representative of [-3, 3]
    grid = np.linspace(-3.0, 3.0, 1000)
    V_grid = np.empty((grid.size, 10), dtype=np.float64)
    V_grid[:, 0] = np.ones(grid.size)
    V_grid[:, 1] = grid
    V_grid[:, 2] = grid * grid
    V_grid[:, 3] = V_grid[:, 2] * grid
    V_grid[:, 4] = np.sin(grid)
    V_grid[:, 5] = np.cos(grid)
    V_grid[:, 6] = np.sin(2.0 * grid)
    V_grid[:, 7] = np.cos(2.0 * grid)
    V_grid[:, 8] = 1.0 / (1.0 + V_grid[:, 2])
    V_grid[:, 9] = np.exp(-V_grid[:, 2] / 4.0)
    
    # RMS of each column
    rms = np.sqrt(np.mean(V_grid ** 2, axis=0))
    # Avoid division by zero
    rms = np.maximum(rms, 1e-12)
    # Scaling factors: we want to scale params so that effective coefficients are reasonable
    # If we scale V columns by 1/rms, then the loss surface is better conditioned
    scale = 1.0 / rms
    
    # Apply scaling to V columns
    V_scaled = V * scale[np.newaxis, :]
    
    return V_scaled @ params
# EVOLVE-BLOCK-END
