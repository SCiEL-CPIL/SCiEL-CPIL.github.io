"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bilinear Chebyshev basis: T_i(x) * T_j(x) for selected pairs (i,j).
    This allows capturing non-polynomial features like |x| or sharp transitions
    more efficiently than a single linear basis by exploiting multiplicative interactions.
    The model is linear in the 10 parameters, ensuring BFGS converges to the global minimum.
    
    We use products T_i(x) * T_j(x) for specific pairs to span a rich function space.
    The 10 basis functions are chosen to cover various degrees and interactions.
    """
    # Compute Chebyshev polynomials up to degree 9
    # Using recurrence: T_0=1, T_1=x, T_{n+1} = 2x*T_n - T_{n-1}
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = x
    for n in range(2, 10):
        T[n, :] = 2.0 * x * T[n-1, :] - T[n-2, :]
    
    # Select 10 basis functions as products T_i(x) * T_j(x)
    # We want to maximize coverage of polynomial degrees 0-9
    # and include some "sharp" features via higher-order interactions
    
    # Pairs (i, j) chosen to span degrees 0 through 9 and capture structure:
    # (0,0) -> T0*T0 = 1 (degree 0)
    # (0,1) -> T1 (degree 1)
    # (0,2) -> T2 (degree 2)
    # (0,3) -> T3 (degree 3)
    # (0,4) -> T4 (degree 4)
    # (1,1) -> T1*T1 = x^2 (degree 2, different from T2)
    # (0,5) -> T5 (degree 5)
    # (1,2) -> T1*T2 (degree 3)
    # (0,6) -> T6 (degree 6)
    # (1,3) -> T1*T3 (degree 4)
    
    # Actually, let's think more carefully. The current approach uses:
    # b0=T0*T0, b1=T0*T1, b2=T0*T2, b3=T0*T3, b4=T0*T4,
    # b5=T1*T1, b6=T1*T2, b7=T1*T3, b8=T2*T2, b9=T1*T4
    
    # Let's try a different selection that might be better conditioned:
    # Use T0*Tk for k=0..5 (6 basis functions for degrees 0-5)
    # And T1*Tk for k=1..4 (4 basis functions for degrees 2-5 with different structure)
    
    # This gives us:
    # b0 = T0*T0 = 1
    # b1 = T0*T1 = x
    # b2 = T0*T2 = 2x^2-1
    # b3 = T0*T3 = 4x^3-3x
    # b4 = T0*T4 = 8x^4-8x^2+1
    # b5 = T0*T5 = 16x^5-20x^3+5x
    # b6 = T1*T1 = x^2
    # b7 = T1*T2 = 2x^3-x
    # b8 = T1*T3 = 4x^4-3x^2
    # b9 = T1*T4 = 8x^5-8x^3+x
    
    # This is actually the same as before but with T5 instead of T2*T2 and T1*T4
    # Let's try yet another combination:
    # b0 = T0*T0
    # b1 = T0*T1
    # b2 = T0*T2
    # b3 = T0*T3
    # b4 = T0*T4
    # b5 = T0*T5
    # b6 = T0*T6
    # b7 = T1*T1
    # b8 = T1*T2
    # b9 = T1*T3
    
    # This gives degrees 0-6 from T0 series, plus some quadratic/cubic/quartic from T1 series
    # This should capture polynomials up to degree 6 well
    
    b0 = T[0, :] * T[0, :]  # 1
    b1 = T[0, :] * T[1, :]  # x
    b2 = T[0, :] * T[2, :]  # 2x^2-1
    b3 = T[0, :] * T[3, :]  # 4x^3-3x
    b4 = T[0, :] * T[4, :]  # 8x^4-8x^2+1
    b5 = T[0, :] * T[5, :]  # 16x^5-20x^3+5x
    b6 = T[0, :] * T[6, :]  # 32x^6-48x^4+18x^2-1
    b7 = T[1, :] * T[1, :]  # x^2
    b8 = T[1, :] * T[2, :]  # 2x^3-x
    b9 = T[1, :] * T[3, :]  # 4x^4-3x^2
    
    # Linear combination of basis functions
    y = (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
         params[4] * b4 + params[5] * b5 + params[6] * b6 + params[7] * b7 +
         params[8] * b8 + params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
