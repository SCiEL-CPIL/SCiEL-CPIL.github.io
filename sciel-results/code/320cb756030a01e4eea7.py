"""Improved optimization approach for the uncertainty inequality task."""
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    We minimize the upper bound on A(f)*A(F(fourier(f))) which relates to C4.
    """
    
    # Precompute factorials and coefficients
    # He_n(0) values for even n: He_0(0)=1, He_4(0)=3, He_8(0)=105, He_12(0)=10395
    # <He_n, He_n> = n!
    # <x^2 He_n, He_n> = (2n+1) * n!
    # <He_n', He_n'> = n * (n-1)! * n = n^2 * (n-1)! ... let me recalculate
    # He_n' = n * He_{n-1}
    # <He_n', He_n'> = n^2 * <He_{n-1}, He_{n-1}> = n^2 * (n-1)!
    
    def objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        
        # Norm squared: a0^2*0! + a4^2*4! + a8^2*8! + a12^2*12!
        norm_sq = a0**2 * 1 + a4**2 * 24 + a8**2 * 40320 + a12**2 * 479001600
        
        if norm_sq < 1e-15:
            return 1e15
        
        # x^2 inner product: <x^2 He_n, He_n> = (2n+1)*n!
        # n=0: 1*1=1, n=4: 9*24=216, n=8: 17*40320=685440, n=12: 25*479001600=11975040000
        x2_inner = a0**2 * 1 + a4**2 * 216 + a8**2 * 685440 + a12**2 * 11975040000
        x2_var = x2_inner / norm_sq
        
        # Derivative norm: <He_n', He_n'> = n^2 * (n-1)!
        # n=4: 16*6=96, n=8: 64*5040=322560, n=12: 144*39916800=5748019200
        deriv_norm_sq = a4**2 * 96 + a8**2 * 322560 + a12**2 * 5748019200
        freq_var = deriv_norm_sq / norm_sq
        
        # Minimize the product of variances (or its square root, same optimum)
        return x2_var * freq_var
    
    # Try multiple initial points and use different optimization methods
    best_result = None
    best_val = np.inf
    
    # Various initial guesses
    initial_guesses = [
        np.array([-1.0, 0.01, 0.0]),
        np.array([-1.0, 0.04, 0.0]),
        np.array([-1.0, 0.0, 0.001]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-1.0, 0.03, 0.0002]),
        np.array([-113.0/100, 1.0/25, 1.0/3240]),
        np.array([-1.0, 0.1, 0.0]),
        np.array([-0.1, 0.01, 0.0]),
        np.array([1.0, -0.01, 0.0]),
    ]
    
    for x0 in initial_guesses:
        try:
            # Try L-BFGS-B
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_result = result.x.copy()
        except:
            pass
        
        try:
            # Try Nelder-Mead
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_result = result.x.copy()
        except:
            pass
    
    if best_result is not None:
        return best_result
    else:
        return np.array([-1.0, 0.04, 0.0])


# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
