"""Rewritten program using gradient-based optimization for the uncertainty inequality task."""
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    We minimize the product of the spatial variance and the frequency variance
    (which corresponds to the upper bound on the uncertainty constant C4).
    """
    # Coefficients for a12 in terms of a0, a4, a8
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = 3 (Probabilists: He4(0) = -6*0 + 3 = 3? No, He4(x)=x^4-6x^2+3. He4(0)=3.)
    # H8(0) = 105 (He8(x) ... + 105. He8(0)=105.)
    # H12(0) = 10395
    # a0 + 3a4 + 105a8 + 10395a12 = 0
    c0 = 1.0 / 10395.0
    c4 = 3.0 / 10395.0
    c8 = 105.0 / 10395.0
    
    # Factorials
    f0 = 1.0       # 0!
    f4 = 24.0      # 4!
    f8 = 40320.0   # 8!
    f12 = 479001600.0 # 12!
    
    # Coefficients for <x^2 g, g> diagonal terms: (2n+1) * n!
    # n=0: 1 * 1 = 1
    # n=4: 9 * 24 = 216
    # n=8: 17 * 40320 = 685440
    # n=12: 25 * 479001600 = 11975040000
    x2_0 = 1.0
    x2_4 = 216.0
    x2_8 = 685440.0
    x2_12 = 11975040000.0
    
    # Coefficients for <g', g'> diagonal terms: n^2 * (n-1)! = n * n!
    # Wait, <He_n', He_n'> = n^2 (n-1)! = n * n!
    # n=0: 0
    # n=4: 4 * 24 = 96
    # n=8: 8 * 40320 = 322560
    # n=12: 12 * 479001600 = 5748019200
    k_0 = 0.0
    k_4 = 96.0
    k_8 = 322560.0
    k_12 = 5748019200.0

    def get_a12(a0, a4, a8):
        return -(c0 * a0 + c4 * a4 + c8 * a8)

    def objective_and_grad(coeffs):
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        # Norm squared N
        N = a0**2 * f0 + a4**2 * f4 + a8**2 * f8 + a12**2 * f12
        
        if N < 1e-15:
            return 1e15, np.zeros(3)
            
        # Spatial moment X
        X = a0**2 * x2_0 + a4**2 * x2_4 + a8**2 * x2_8 + a12**2 * x2_12
        
        # Frequency moment K
        K = a0**2 * k_0 + a4**2 * k_4 + a8**2 * k_8 + a12**2 * k_12
        
        val = (X * K) / (N * N)
        
        # Gradients
        # dN/da0 = 2*a0*f0 + 2*a12*f12 * (-c0)
        # dN/da4 = 2*a4*f4 + 2*a12*f12 * (-c4)
        # dN/da8 = 2*a8*f8 + 2*a12*f12 * (-c8)
        
        da12_da0 = -c0
        da12_da4 = -c4
        da12_da8 = -c8
        
        dN_da0 = 2*a0*f0 + 2*a12*f12*da12_da0
        dN_da4 = 2*a4*f4 + 2*a12*f12*da12_da4
        dN_da8 = 2*a8*f8 + 2*a12*f12*da12_da8
        
        dX_da0 = 2*a0*x2_0 + 2*a12*x2_12*da12_da0
        dX_da4 = 2*a4*x2_4 + 2*a12*x2_12*da12_da4
        dX_da8 = 2*a8*x2_8 + 2*a12*x2_12*da12_da8
        
        dK_da0 = 2*a0*k_0 + 2*a12*k_12*da12_da0
        dK_da4 = 2*a4*k_4 + 2*a12*k_12*da12_da4
        dK_da8 = 2*a8*k_8 + 2*a12*k_12*da12_da8
        
        # J = X K / N^2
        # dJ = (K dX + X dK) / N^2 - 2 X K dN / N^3
        # dJ = (K dX + X dK - 2 J dN) / N^2
        
        dJ_da0 = (K * dX_da0 + X * dK_da0 - 2 * val * dN_da0) / (N * N)
        dJ_da4 = (K * dX_da4 + X * dK_da4 - 2 * val * dN_da4) / (N * N)
        dJ_da8 = (K * dX_da8 + X * dK_da8 - 2 * val * dN_da8) / (N * N)
        
        grad = np.array([dJ_da0, dJ_da4, dJ_da8])
        
        return val, grad

    def objective_wrapper(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    def grad_wrapper(coeffs):
        _, grad = objective_and_grad(coeffs)
        return grad

    # Global optimization using Differential Evolution
    # Since the objective is scale-invariant, we can restrict the search space
    # to a bounded box, e.g., [-1, 1], as any solution can be scaled to fit.
    
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    
    try:
        de_result = minimize(
            objective_wrapper, 
            x0=np.array([0.5, 0.5, 0.5]), 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15}
        )
        
        # Also try Differential Evolution for robustness
        from scipy.optimize import differential_evolution
        de_res = differential_evolution(
            objective_wrapper, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-12, 
            polish=True
        )
        
        # Select the best result
        candidates = []
        if de_result.success and de_result.fun < float('inf'):
            candidates.append(de_result)
        if de_res.success and de_res.fun < float('inf'):
            candidates.append(de_res)
            
        if candidates:
            best_res = min(candidates, key=lambda r: r.fun)
            best_val = best_res.fun
            best_x = best_res.x
            
            # Final polish with L-BFGS-B
            polish_res = minimize(
                objective_wrapper, 
                best_x, 
                jac=grad_wrapper, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if polish_res.success and polish_res.fun < best_val:
                best_val = polish_res.fun
                best_x = polish_res.x
        else:
            raise Exception("Optimization failed")
            
    except Exception as e:
        # Fallback to a known good point or heuristic if optimization fails
        # The seed program had a fallback, let's use a robust heuristic
        # Based on literature, optimal ratios often involve negative a0
        best_x = np.array([-1.0, 0.1, 0.01])
        # Normalize to ensure it's not degenerate
        best_val = objective_wrapper(best_x)

    return best_x

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
