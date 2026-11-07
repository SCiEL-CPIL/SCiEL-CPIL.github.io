import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    2D Chebyshev polynomial basis for nonlinear oscillator.
    
    Chebyshev polynomials are orthogonal on [-1, 1], providing better
    numerical conditioning for the BFGS optimizer compared to monomials.
    This basis allows for flexible approximation of smooth nonlinear dynamics.
    """
    # Scale inputs to approximately [-1, 1]
    # x range approx [-0.57, 0.53] -> center 0, radius 0.55
    # v range approx [-0.18, 0.28] -> center 0.05, radius 0.23
    
    x_c = 0.0
    x_r = 0.55
    v_c = 0.05
    v_r = 0.23
    
    x_s = (x - x_c) / x_r
    v_s = (v - v_c) / v_r
    
    # Clip to [-1, 1] to ensure arccos is valid and stable
    x_s = np.clip(x_s, -1.0, 1.0)
    v_s = np.clip(v_s, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_3 for x and v
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x_s
    T2_x = 2.0 * x_s * x_s - 1.0
    T3_x = 4.0 * x_s * x_s * x_s - 3.0 * x_s
    
    T0_v = np.ones_like(v)
    T1_v = v_s
    T2_v = 2.0 * v_s * v_s - 1.0
    T3_v = 4.0 * v_s * v_s * v_s - 3.0 * v_s
    
    # Construct 10 basis functions (2D Chebyshev terms)
    b0 = T0_x * T0_v  # 1
    b1 = T1_x * T0_v  # x_s
    b2 = T0_x * T1_v  # v_s
    b3 = T2_x * T0_v  # 2x_s^2 - 1
    b4 = T0_x * T2_v  # 2v_s^2 - 1
    b5 = T1_x * T1_v  # x_s * v_s
    b6 = T3_x * T0_v  # 4x_s^3 - 3x_s
    b7 = T0_x * T3_v  # 4v_s^3 - 3v_s
    b8 = T2_x * T1_v  # (2x_s^2 - 1) * v_s
    b9 = T1_x * T2_v  # x_s * (2v_s^2 - 1)
    
    # Linear combination
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
