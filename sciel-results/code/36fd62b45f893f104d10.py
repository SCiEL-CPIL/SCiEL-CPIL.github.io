"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for better conditioning
    # x is in [-3, 3], so u = x / 3 is in [-1, 1]
    u = x / 3.0
    
    # Calculate Chebyshev Polynomials T_0 to T_9
    # Recurrence relation: T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
    
    t0 = np.ones_like(u)  # T_0
    t1 = u                # T_1
    
    # Build terms T_0 ... T_9
    term_0 = t0
    term_1 = t1
    
    # T_2
    t2 = 2.0 * u * t1 - t0
    term_2 = t2
    
    # T_3
    t3 = 2.0 * u * t2 - t1
    term_3 = t3
    
    # T_4
    t4 = 2.0 * u * t3 - t2
    term_4 = t4
    
    # T_5
    t5 = 2.0 * u * t4 - t3
    term_5 = t5
    
    # T_6
    t6 = 2.0 * u * t5 - t4
    term_6 = t6
    
    # T_7
    t7 = 2.0 * u * t6 - t5
    term_7 = t7
    
    # T_8
    t8 = 2.0 * u * t7 - t6
    term_8 = t8
    
    # T_9
    t9 = 2.0 * u * t8 - t7
    term_9 = t9
    
    # Combine with params
    # params length is 10
    c = params
    
    return (c[0] * term_0 + 
            c[1] * term_1 + 
            c[2] * term_2 + 
            c[3] * term_3 + 
            c[4] * term_4 + 
            c[5] * term_5 + 
            c[6] * term_6 + 
            c[7] * term_7 + 
            c[8] * term_8 + 
            c[9] * term_9)

# EVOLVE-BLOCK-END
