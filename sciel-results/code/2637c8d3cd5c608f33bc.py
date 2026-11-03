import numpy as np
from scipy.optimize import minimize

def construct():
    # Coefficients for H0, H4, H8
    # H12 is derived
    
    # Probabilists' Hermite polynomials He_n
    # psi_n(x) = (1 / (pi^0.25 * sqrt(n!))) * exp(-x^2/2) * He_n(x)
    # psi_n(0) = (1 / (pi^0.25 * sqrt(n!))) * He_n(0)
    
    # He_n(0) for even n=2k is (-1)^k * (2k-1)!!
    # (2k-1)!! = (2k)! / (2^k * k!)
    
    def he_even_zero(n):
        if n % 2 != 0:
            return 0
        k = n // 2
        # (2k-1)!!
        double_fact = 1
        for i in range(1, k+1):
            double_fact *= (2*i - 1)
        return ((-1)**k) * double_fact

    def psi_zero(n):
        if n % 2 != 0:
            return 0
        k = n // 2
        # factorial(n)
        import math
        fact_n = math.factorial(n)
        val = he_even_zero(n) / (np.pi**0.25 * np.sqrt(fact_n))
        return val

    K0 = psi_zero(0)
    K4 = psi_zero(4)
    K8 = psi_zero(8)
    K12 = psi_zero(12)
    
    # Costs d_n = n + 0.5
    d0 = 0.5
    d4 = 4.5
    d8 = 8.5
    d12 = 12.5
    
    # Objective function to minimize
    # Variables: c4, c8, c12
    # c0 is determined by constraint: c0*K0 + c4*K4 + c8*K8 + c12*K12 = 0
    # c0 = -(c4*K4 + c8*K8 + c12*K12) / K0
    
    def objective(vars):
        c4, c8, c12 = vars
        c0 = -(c4*K4 + c8*K8 + c12*K12) / K0
        
        # Normalize? 
        # The Rayleigh quotient is homogeneous of degree 2.
        # J = (d0*c0^2 + d4*c4^2 + d8*c8^2 + d12*c12^2) / (c0^2 + c4^2 + c8^2 + c12^2)
        num = d0*c0**2 + d4*c4**2 + d8*c8**2 + d12*c12**2
        den = c0**2 + c4**2 + c8**2 + c12**2
        if den == 0:
            return 1e10
        return num / den

    # Constraints
    # We just need to find the direction that minimizes the Rayleigh quotient.
    # The magnitude doesn't matter for the quotient.
    # We can fix c12 = 1 or something, or just let the optimizer find the direction.
    # However, minimize with SLSQP might struggle with scale invariance.
    # Let's add a constraint c4^2 + c8^2 + c12^2 + c0^2 = 1 to fix the scale.
    
    def constraint_norm(vars):
        c4, c8, c12 = vars
        c0 = -(c4*K4 + c8*K8 + c12*K12) / K0
        return c0**2 + c4**2 + c8**2 + c12**2 - 1

    # Initial guess
    # Start with seed-like values or random
    # Seed: c0=-1.13, c4=0.04, c8=0.0003
    # These are not normalized.
    # Let's try a random start.
    np.random.seed(42)
    x0 = np.random.randn(3) * 0.1
    
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=[{'type': 'eq', 'fun': constraint_norm}])
    
    if res.success:
        c4, c8, c12 = res.x
        c0 = -(c4*K4 + c8*K8 + c12*K12) / K0
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

# EVOLVE-BLOCK-END
