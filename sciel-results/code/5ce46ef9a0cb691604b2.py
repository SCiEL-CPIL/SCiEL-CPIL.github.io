"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a normalized quadratic polynomial basis.
    
    This approach replaces fixed-center biological terms (Monod, Gaussian) with
    a flexible polynomial expansion on normalized inputs. This allows the
    optimizer to determine the optimal shape, centers, and interactions
    without being constrained to specific mechanistic assumptions.
    
    Inputs are normalized to [0, 1] based on the training range to improve
    numerical stability for the optimizer.
    """
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5], s: [1, 5], temp: [1, 40], pH: [2, 12]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Basis functions:
    # 0: Constant
    # 1-4: Linear terms
    # 5-8: Quadratic terms
    # 9: Interaction term (Temperature * pH)
    
    term0 = np.ones_like(b)
    term1 = b_n
    term2 = s_n
    term3 = t_n
    term4 = p_n
    term5 = b_n * b_n
    term6 = s_n * s_n
    term7 = t_n * t_n
    term8 = p_n * p_n
    term9 = t_n * p_n
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
