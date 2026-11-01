"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard
START_TIME = time.time()
DEADLINE = 180.0  # seconds, with safety margin before 210s kill

def hermite_polynomials(x, n_max=12):
    """Compute Hermite polynomials H_0 through H_{n_max} at points x."""
    H = np.zeros((n_max + 1, len(x)))
    H[0, :] = 1.0
    if n_max >= 1:
        H[1, :] = 2 * x
    for n in range(2, n_max + 1):
        H[n, :] = 2 * x * H[n-1, :] - 2 * (n-1) * H[n-2, :]
    return H

def construct():
    # Check deadline
    if time.time() - START_TIME > DEADLINE:
        return np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    # Define the objective function
    def objective(params):
        a0, a4, a8 = params
        
        # Compute a12 from P(0) = 0
        # H0(0) = 1, H4(0) = 3, H8(0) = 105, H12(0) = 10395
        a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
        
        # Evaluate g(x) on a grid
        x = np.linspace(-10, 10, 4001)
        H = hermite_polynomials(x, 12)
        
        P = a0 * H[0] + a4 * H[4] + a8 * H[8] + a12 * H[12]
        g = P * np.exp(-x**2 / 2.0)
        
        # Compute L1 norm (area under |g|)
        dx = x[1] - x[0]
        l1 = np.sum(np.abs(g)) * dx
        
        # Since Fourier-invariant, A(g)*A(F(g)) = l1^2
        # But let's also consider other measures
        # Return the square of L1 norm as proxy for uncertainty product
        return l1 ** 2
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    # Use Nelder-Mead optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if time.time() - START_TIME > DEADLINE:
            return x0
        best_params = result.x
    except Exception:
        best_params = x0
    
    # Check deadline before returning
    if time.time() - START_TIME > DEADLINE:
        return x0
    
    return best_params
# EVOLVE-BLOCK-END
