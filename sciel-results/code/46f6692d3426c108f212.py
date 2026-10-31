import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin
    
    # Grid setup
    # The functions decay as e^{-x^2/2}. 15 sigma is e^{-112}, negligible.
    # 10 sigma is e^{-50}, ~1e-22.
    # Let's use [-12, 12] with high resolution.
    x = np.linspace(-12, 12, 20000)
    dx = x[1] - x[0]
    
    # Precompute Hermite functions psi_0, psi_4, psi_8, psi_12
    def get_psi(n):
        # psi_n(x) = (2^n n! sqrt(pi))^{-1/2} e^{-x^2/2} H_n(x)
        norm = np.sqrt(2**n * factorial(n, exact=True) * np.sqrt(np.pi))
        H_n = eval_hermite(n, x)
        return (1.0 / norm) * np.exp(-x**2 / 2) * H_n
        
    psi_0 = get_psi(0)
    psi_4 = get_psi(4)
    psi_8 = get_psi(8)
    psi_12 = get_psi(12)
    
    # Values at 0 for constraint
    # psi_n(0) = (2^n n! sqrt(pi))^{-1/2} H_n(0)
    # H_0(0)=1, H_4(0)=12, H_8(0)=1680, H_12(0)=665280
    # Using the precomputed arrays, find index closest to 0
    idx_0 = np.argmin(np.abs(x))
    p0_0 = psi_0[idx_0]
    p0_4 = psi_4[idx_0]
    p0_8 = psi_8[idx_0]
    p0_12 = psi_12[idx_0]
    
    def objective(params):
        c0, c4, c8 = params
        # Calculate c12 such that g(0) = 0
        # c0*p0_0 + c4*p0_4 + c8*p0_8 + c12*p0_12 = 0
        c12 = -(c0 * p0_0 + c4 * p0_4 + c8 * p0_8) / p0_12
        
        g = c0 * psi_0 + c4 * psi_4 + c8 * psi_8 + c12 * psi_12
        
        # Calculate L2 norm squared and L1 norm
        l2_sq = np.sum(g**2) * dx
        if l2_sq < 1e-10:
            return 1e10
        l1 = np.sum(np.abs(g)) * dx
        
        # We want to minimize (L1/L2)^2
        return (l1 / np.sqrt(l2_sq))**2

    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Optimization
    # Use L-BFGS-B for robustness
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        if time.time() > deadline - 10:
             # If close to deadline, stick with best found or seed
             # But minimize returns best found so far in res.x if it converged or stopped
             # We should check if res.success or just use res.x
             pass
             
        best_params = res.x
    except Exception:
        best_params = x0

    # Final check: ensure we return the best result found within time
    # The objective returns the score (upper bound on C4? or the value to minimize?)
    # Task says "score is the upper bound on C4" and "lower is better".
    # So the value we minimize IS the score.
    
    return np.array(best_params)
