"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Basis Expansion Model for E. coli growth.
    
    Strategy:
    1. Use a linear combination of basis functions. This makes the problem convex 
       with respect to the parameters, ensuring BFGS finds the global minimum of 
       the training MSE within this function class.
    2. Basis functions:
       - Substrate: Monod saturation s/(1+s) and linear s.
       - Temperature: Sum of 4 Gaussian kernels centered at 30, 37, 41, 44.
       - pH: Sum of 3 Gaussian kernels centered at 6.5, 7.0, 7.5.
       - Density (b): Exponential decay exp(-0.5*b) and linear b.
       - Constant bias.
    3. The fixed centers and widths allow the fitter to shape the response curve 
       by adjusting the weights (params), capturing asymmetry and plateaus 
       without non-convex structural parameters.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Substrate Terms
    # Monod-like saturation
    f_s_mono = s / (1.0 + s)
    # Linear substrate (for low concentration regime)
    f_s_lin = s
    
    # 2. Temperature Terms (Gaussian Kernels)
    # E. coli optimal growth is ~37C, but can grow 15-45C.
    # We use a spread of Gaussians to allow flexible shaping.
    # Widths are fixed to ~2-3 degrees to allow local adjustments.
    T1 = np.exp(-((temp - 30.0) ** 2) / (2 * 2.0**2))
    T2 = np.exp(-((temp - 37.0) ** 2) / (2 * 2.0**2))
    T3 = np.exp(-((temp - 41.0) ** 2) / (2 * 2.0**2))
    T4 = np.exp(-((temp - 44.0) ** 2) / (2 * 2.0**2))
    
    # 3. pH Terms (Gaussian Kernels)
    # E. coli optimal pH is ~7.0, range 6.0-8.0.
    P1 = np.exp(-((pH - 6.5) ** 2) / (2 * 0.5**2))
    P2 = np.exp(-((pH - 7.0) ** 2) / (2 * 0.5**2))
    P3 = np.exp(-((pH - 7.5) ** 2) / (2 * 0.5**2))
    
    # 4. Density Terms (b)
    # Inhibition at high density
    f_b_exp = np.exp(-0.5 * b)
    f_b_lin = b
    
    # 5. Construct Linear Combination
    # p0: Bias
    # p1: Substrate Monod
    # p2: Substrate Linear
    # p3-p6: Temperature Gaussians
    # p7-p9: pH Gaussians
    # We need 10 params total. 
    # Current count: 1 (bias) + 2 (s) + 4 (T) + 3 (P) = 10. 
    # This leaves no room for explicit density terms if we use all 4 T and 3 P.
    # However, density inhibition is crucial. 
    # Let's adjust the basis set to ensure density is included.
    # 
    # Revised Basis Set (10 terms):
    # 1. Bias (p0)
    # 2. Substrate Monod (p1)
    # 3. Substrate Linear (p2)
    # 4. Temp Gaussian 37C (p3) - Primary optimum
    # 5. Temp Gaussian 30C (p4) - Lower range
    # 6. Temp Gaussian 44C (p5) - Upper range
    # 7. pH Gaussian 7.0 (p6) - Primary optimum
    # 8. pH Gaussian 6.5 (p7) - Lower range
    # 9. pH Gaussian 7.5 (p8) - Upper range
    # 10. Density Exponential (p9)
    # 
    # This drops one Temp Gaussian (41C) to make room for Density. 
    # The 30/37/44 spread should cover the necessary range.
    
    # Recalculate with selected bases
    T_opt = np.exp(-((temp - 37.0) ** 2) / (2 * 2.0**2))
    T_low = np.exp(-((temp - 30.0) ** 2) / (2 * 2.0**2))
    T_high = np.exp(-((temp - 44.0) ** 2) / (2 * 2.0**2))
    
    P_opt = np.exp(-((pH - 7.0) ** 2) / (2 * 0.5**2))
    P_low = np.exp(-((pH - 6.5) ** 2) / (2 * 0.5**2))
    P_high = np.exp(-((pH - 7.5) ** 2) / (2 * 0.5**2))
    
    b_inh = np.exp(-0.5 * b)
    
    y = (p0 
         + p1 * f_s_mono 
         + p2 * f_s_lin 
         + p3 * T_opt 
         + p4 * T_low 
         + p5 * T_high 
         + p6 * P_opt 
         + p7 * P_low 
         + p8 * P_high 
         + p9 * b_inh)
         
    return y
# EVOLVE-BLOCK-END
