"""Improved program using global optimization on normalized coefficients."""
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    We minimize the product of the spatial variance and the frequency variance
    (which corresponds to the upper bound on the uncertainty constant C4).
    """
    # Factorials
    f0 = 1.0
    f4 = 24.0
    f8 = 40320.0
    f12 = 479001600.0
    
    # Square roots of factorials for normalization
    sf0 = np.sqrt(f0)
    sf4 = np.sqrt(f4)
    sf8 = np.sqrt(f8)
    sf12 = np.sqrt(f12)
    
    # Coefficients for the constraint g(0)=0: a0 + 3a4 + 105a8 + 10395a12 = 0
    # In terms of normalized variables u_i = a_i * sqrt(f_i):
    # a_i = u_i / sqrt(f_i)
    # u0/sf0 + 3*u4/sf4 + 105*u8/sf8 + 10395*u12/sf12 = 0
    # u12 = - (sf12 / 10395) * (u0/sf0 + 3*u4/sf4 + 105*u8/sf8)
    
    c_u0 = 1.0 / sf0
    c_u4 = 3.0 / sf4
    c_u8 = 105.0 / sf8
    c_u12_scale = sf12 / 10395.0
    
    # Diagonal terms for X = <x^2 g, g> / N
    # Original: x2_n = (2n+1) n!
    # Normalized contribution to X: (u_n^2 / f_n) * x2_n = u_n^2 * (2n+1)
    x2_norm_0 = 1.0 * 1.0  # (2*0+1) * (0! / 0! * 1/1) -> Wait.
    # Let's re-derive carefully.
    # N = sum a_i^2 f_i = sum u_i^2
    # X = sum a_i^2 x2_i = sum (u_i^2 / f_i) * x2_i
    # x2_i = (2i+1) f_i
    # So term is u_i^2 * (2i+1)
    
    x2_0 = 1.0
    x2_4 = 9.0
    x2_8 = 17.0
    x2_12 = 25.0
    
    # Diagonal terms for K = <g', g'> / N
    # Original: k_n = n n!
    # Normalized contribution to K: (u_n^2 / f_n) * k_n = u_n^2 * n
    k_0 = 0.0
    k_4 = 4.0
    k_8 = 8.0
    k_12 = 12.0
    
    def get_u12(u0, u4, u8):
        return -c_u12_scale * (c_u0 * u0 + c_u4 * u4 + c_u8 * u8)
        
    def objective(u):
        u0, u4, u8 = u
        u12 = get_u12(u0, u4, u8)
        
        # Norm squared N (in normalized variables, this is just sum of squares)
        N = u0**2 + u4**2 + u8**2 + u12**2
        
        if N < 1e-15:
            return 1e15
            
        # Spatial moment X
        X = u0**2 * x2_0 + u4**2 * x2_4 + u8**2 * x2_8 + u12**2 * x2_12
        
        # Frequency moment K
        K = u0**2 * k_0 + u4**2 * k_4 + u8**2 * k_8 + u12**2 * k_12
        
        val = (X * K) / (N * N)
        return val

    # Bounds for differential evolution
    # The variables are normalized, so O(1) is a good scale.
    # We can use a wide range to ensure global coverage.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Run global optimization
    # maxiter and popsize control the thoroughness.
    # 180s limit is generous for 3D global opt.
    res = differential_evolution(objective, bounds, maxiter=1000, popsize=30, 
                                  tol=1e-12, seed=42, polish=True)
    
    if not res.success:
        # Fallback to a reasonable guess if DE fails (unlikely)
        best_u = np.array([-1.0, 0.1, 0.01])
    else:
        best_u = res.x
        
    u0, u4, u8 = best_u
    
    # Convert back to original coefficients a_i = u_i / sqrt(f_i)
    a0 = u0 / sf0
    a4 = u4 / sf4
    a8 = u8 / sf8
    
    return np.array([a0, a4, a8])

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
