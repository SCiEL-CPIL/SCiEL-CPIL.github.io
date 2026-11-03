"""Rewritten program using trust-constr interior-point optimization for the uncertainty inequality task."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is determined by the constraint g(0) = 0.
    We use a trust-constr interior-point method to minimize the uncertainty product.
    """
    # Constants for Hermite polynomials at x=0 (Probabilists)
    # H_n(x) = n! sum_{k=0}^{floor(n/2)} (-1)^k (2x)^(n-2k) / (k! (n-2k)!)
    # H_0(0) = 1
    # H_4(0) = 3
    # H_8(0) = 105
    # H_12(0) = 10395
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 10395.0

    # Norms n!
    f0 = 1.0
    f4 = 24.0
    f8 = 40320.0
    f12 = 479001600.0

    # Spatial moments <x^2 H_n H_n> = (2n+1) n!
    x2_0 = 1.0 * f0
    x2_4 = 9.0 * f4
    x2_8 = 17.0 * f8
    x2_12 = 25.0 * f12

    # Frequency moments <H_n' H_n'> = n (n-1)! * n = n n! ? 
    # He_n' = n He_{n-1}. <He_n', He_n'> = n^2 <He_{n-1}, He_{n-1}> = n^2 (n-1)! = n n!.
    k_0 = 0.0
    k_4 = 4.0 * f4
    k_8 = 8.0 * f8
    k_12 = 12.0 * f12

    # Vectors for quadratic forms
    # Coeff vector a = [a0, a4, a8, a12]
    
    def objective(a):
        a0, a4, a8, a12 = a
        N = a0**2 * f0 + a4**2 * f4 + a8**2 * f8 + a12**2 * f12
        if N < 1e-15:
            return 1e15
        X = a0**2 * x2_0 + a4**2 * x2_4 + a8**2 * x2_8 + a12**2 * x2_12
        K = a0**2 * k_0 + a4**2 * k_4 + a8**2 * k_8 + a12**2 * k_12
        return (X * K) / (N * N)

    def gradient(a):
        a0, a4, a8, a12 = a
        N = a0**2 * f0 + a4**2 * f4 + a8**2 * f8 + a12**2 * f12
        if N < 1e-15:
            return np.zeros(4)
        X = a0**2 * x2_0 + a4**2 * x2_4 + a8**2 * x2_8 + a12**2 * x2_12
        K = a0**2 * k_0 + a4**2 * k_4 + a8**2 * k_8 + a12**2 * k_12
        val = (X * K) / (N * N)
        
        dN = np.array([2*a0*f0, 2*a4*f4, 2*a8*f8, 2*a12*f12])
        dX = np.array([2*a0*x2_0, 2*a4*x2_4, 2*a8*x2_8, 2*a12*x2_12])
        dK = np.array([2*a0*k_0, 2*a4*k_4, 2*a8*k_8, 2*a12*k_12])
        
        # dJ = (K dX + X dK - 2 val dN) / N^2
        grad = (K * dX + X * dK - 2 * val * dN) / (N * N)
        return grad

    # Constraint: a0 + 3a4 + 105a8 + 10395a12 = 0
    # This is a linear constraint A @ a = 0
    A_con = np.array([[h0_0, h4_0, h8_0, h12_0]])
    constraint = LinearConstraint(A_con, 0, 0)

    # Multi-start optimization
    best_val = float('inf')
    best_x = None
    
    # Initial guesses (4 coefficients)
    # We can start with random points on the constraint hyperplane
    # or just random points and let the solver project them.
    starts = [
        np.array([1.0, -0.1, 0.01, 0.0]),
        np.array([0.0, 1.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 1.0, -0.1]),
        np.array([1.0, 1.0, 1.0, -1.0]),
        np.array([-1.0, 0.5, -0.25, 1.0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([-0.1, -0.2, -0.3, -0.4]),
        np.array([1.0, -1.0, 1.0, -1.0]),
    ]
    
    for start in starts:
        try:
            res = minimize(objective, start, jac=gradient, method='trust-constr',
                           constraints=[constraint],
                           options={'maxiter': 1000, 'gtol': 1e-12, 'verbose': 0})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback: project a random point onto the constraint and use it
        # Or use the previous hardcoded fallback
        best_x = np.array([-113/100, 1/25, 1/3240, 0.0])
        # Ensure constraint is satisfied for fallback
        # a12 = -(a0 + 3a4 + 105a8)/10395
        a0, a4, a8 = best_x[0], best_x[1], best_x[2]
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        best_x[3] = a12

    return best_x[:3]

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
