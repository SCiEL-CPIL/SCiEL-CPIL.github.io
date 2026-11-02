"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 for a test function of the form
    f(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    where a12 is determined by the constraint f(0) = 0.
    
    The Hermite polynomials are the physicists' Hermite polynomials:
    H0(x) = 1
    H4(x) = 16x^4 - 48x^2 + 12
    H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    H12(x) = 4096x^12 - 73728x^10 + 580608x^8 - 2419200x^6 + 5443200x^4 - 5443200x^2 + 2032128
    
    The Fourier transform of H_n(x) * exp(-x^2/2) is (-i)^n H_n(x) * exp(-x^2/2).
    For n=0,4,8,12, (-i)^n = 1, so the function is Fourier-invariant.
    
    The constraint g(0)=0 means:
    a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    H0(0) = 1
    H4(0) = 12
    H8(0) = 1680
    H12(0) = 2032128
    
    So: a0 + 12*a4 + 1680*a8 + 2032128*a12 = 0
    a12 = -(a0 + 12*a4 + 1680*a8) / 2032128
    
    The quantity to minimize is the upper bound on A(f)*A(Fourier(f)), which for
    Fourier-invariant functions is related to the L1 norm squared.
    """
    # Compute a12 from constraint f(0) = 0
    a12 = -(a0 + 12*a4 + 1680*a8) / 2032128.0
    
    # Define the Hermite polynomials
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    
    def H12(x):
        return 4096*x**12 - 73728*x**10 + 580608*x**8 - 2419200*x**6 + 5443200*x**4 - 5443200*x**2 + 2032128
    
    # Use numerical integration on a fine grid to compute the L1 norm
    # The bound is related to the product of L1 norms in position and frequency
    # Since the function is Fourier-invariant, this is (L1 norm)^2
    
    # Use a grid that captures the function well
    x = np.linspace(-15, 15, 5000)
    dx = x[1] - x[0]
    
    f = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * np.exp(-x**2/2)
    
    # Compute L1 norm using trapezoidal rule
    l1_norm = np.trapz(np.abs(f), x)
    
    # The bound on A(f)*A(F) is proportional to l1_norm^2
    bound = l1_norm**2
    
    return bound


def construct():
    start_time = time.time()
    deadline = 210.0  # seconds - evaluator kills after this
    
    # Initial guess from the seed
    x0 = np.array([-1.13, 0.04, 0.00030864])
    
    best_x = x0.copy()
    best_val = _compute_bound(x0[0], x0[1], x0[2])
    
    # Try multiple optimization strategies
    try:
        # Strategy 1: Nelder-Mead from seed
        result = minimize(_compute_bound, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        
        elapsed = time.time() - start_time
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
        
        if elapsed > deadline - 30:  # Safety margin
            return best_x
        
        # Strategy 2: Try different initial points
        initial_points = [
            np.array([-1.0, 0.05, 0.0003]),
            np.array([-1.2, 0.03, 0.0004]),
            np.array([-0.9, 0.06, 0.0002]),
            np.array([-1.5, 0.02, 0.0005]),
            np.array([-0.8, 0.07, 0.0001]),
        ]
        
        for ip in initial_points:
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                break
            
            try:
                result = minimize(_compute_bound, ip, method='Nelder-Mead', 
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Strategy 3: Polish with L-BFGS-B if we have a good starting point
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            try:
                result = minimize(_compute_bound, best_x, method='L-BFGS-B', 
                                 options={'maxiter': 1000, 'gtol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Strategy 4: Final high-precision polish
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            try:
                result = minimize(_compute_bound, best_x, method='Nelder-Mead', 
                                 options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
                
    except Exception:
        pass
    
    return best_x

# EVOLVE-BLOCK-END
