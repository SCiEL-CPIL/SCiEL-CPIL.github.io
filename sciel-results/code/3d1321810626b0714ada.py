import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map inputs to [-1, 1] for Chebyshev polynomials
    # Training ranges from problem description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Avoid division by zero if ranges are degenerate, though they aren't here
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    t_x = (2.0 * x - (x_max + x_min)) / x_range
    t_v = (2.0 * v - (v_max + v_min)) / v_range
    
    # Clip to [-1, 1] to ensure stability if test data is slightly out of bounds
    # Chebyshev polynomials grow rapidly outside [-1, 1]
    t_x = np.clip(t_x, -1.0, 1.0)
    t_v = np.clip(t_v, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_n(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
    T0_x = np.ones_like(t_x)
    T1_x = t_x
    T2_x = 2.0 * t_x**2 - 1.0
    T3_x = 4.0 * t_x**3 - 3.0 * t_x
    T4_x = 8.0 * t_x**4 - 8.0 * t_x**2 + 1.0
    
    T0_v = np.ones_like(t_v)
    T1_v = t_v
    T2_v = 2.0 * t_v**2 - 1.0
    T3_v = 4.0 * t_v**3 - 3.0 * t_v
    T4_v = 8.0 * t_v**4 - 8.0 * t_v**2 + 1.0
    
    # Basis functions
    # We have 10 params.
    # Let's allocate them to capture the likely physics:
    # Duffing oscillator: a ~ x + x^3 + v
    # Chebyshev T1, T3 capture odd powers well. T0 is bias.
    # T2, T4 capture even powers (might be 0 but good for generalization).
    # Cross term T1_x * T1_v for interaction.
    
    # p0: Bias (T0_x)
    # p1: x linear (T1_x)
    # p2: x quadratic (T2_x)
    # p3: x cubic (T3_x)
    # p4: x quartic (T4_x)
    # p5: v linear (T1_v)
    # p6: v quadratic (T2_v)
    # p7: v cubic (T3_v)
    # p8: v quartic (T4_v)
    # p9: x*v interaction (T1_x * T1_v)
    
    return (params[0] * T0_x + 
            params[1] * T1_x + 
            params[2] * T2_x + 
            params[3] * T3_x + 
            params[4] * T4_x + 
            params[5] * T1_v + 
            params[6] * T2_v + 
            params[7] * T3_v + 
            params[8] * T4_v + 
            params[9] * (T1_x * T1_v))
