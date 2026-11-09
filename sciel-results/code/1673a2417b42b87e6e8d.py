"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a feature-engineered linear model.
    
    This model uses normalized inputs and includes:
    - Linear terms for all inputs.
    - Monod saturation term for substrate.
    - Quadratic terms for temperature and pH (capturing optimal ranges).
    - Key interaction terms (T*pH, S*T, B*T).
    
    Parameters are fitted by the evaluator using BFGS.
    """
    # Normalize inputs to improve optimization stability (BFGS prefers O(1) features)
    # Ranges estimated from task description:
    # b: [1, 5] -> scale 4
    # s: [1, 50] -> scale 49 (using 48 for safety)
    # temp: [1, 40] -> scale 39
    # pH: [2, 12] -> scale 10
    
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 48.0
    t_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Monod term for substrate saturation
    # Using s_n in the monod term to keep it bounded
    s_monod = s_n / (1.0 + s_n)
    
    # Construct features
    # 1. Intercept
    # 2. Linear b
    # 3. Monod s
    # 4. Linear t
    # 5. Linear p
    # 6. Quadratic t
    # 7. Quadratic p
    # 8. Interaction t * p
    # 9. Interaction s * t
    # 10. Interaction b * t
    
    return (params[0] * 1.0 +
            params[1] * b_n +
            params[2] * s_monod +
            params[3] * t_n +
            params[4] * p_n +
            params[5] * (t_n ** 2) +
            params[6] * (p_n ** 2) +
            params[7] * (t_n * p_n) +
            params[8] * (s_n * t_n) +
            params[9] * (b_n * t_n))
# EVOLVE-BLOCK-END
