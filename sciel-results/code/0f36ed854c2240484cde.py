"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as an unconstrained 4th-order multivariate
    polynomial in normalized inputs.  Exactly 10 monomials are used so that
    scipy.optimize.minimize can fit the 10 coefficients.

    Inputs:
        b: population density
        s: substrate concentration
        temp: temperature
        pH: pH level
        params: array of 10 fitted constants

    The inputs are normalized to roughly [0, 1] based on training ranges:
        b: [1, 4.998] -> (b - 1) / 3.998
        s: [1, 4.999] -> (s - 1) / 3.999
        temp: [1.002, 39.98] -> (temp - 1.002) / 38.978
        pH: [2, 12] -> (pH - 2) / 10

    We choose 10 monomials from a 4th-order multivariate polynomial basis to
    capture the main effects, the most likely pairwise interactions, and a few
    key higher-order dependencies within the 10-parameter budget.
    """
    # Normalize inputs to approximately [0, 1]
    b_norm = (b - 1.0) / 3.998
    s_norm = (s - 1.0) / 3.999
    temp_norm = (temp - 1.002) / 38.978
    ph_norm = (pH - 2.0) / 10.0

    # Mild clipping to keep values sane if a row is slightly outside training range
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)

    # Construct 10 monomials from a 4th-order multivariate polynomial:
    # 1. 1 (constant)
    # 2. b
    # 3. s
    # 4. temp
    # 5. pH
    # 6. b * s
    # 7. s * temp
    # 8. temp * pH
    # 9. s * temp * pH   (three-way interaction)
    # 10. temp^2          (quadratic term for non-monotonic temperature response)
    t1 = np.ones_like(b_norm)
    t2 = b_norm
    t3 = s_norm
    t4 = temp_norm
    t5 = ph_norm
    t6 = b_norm * s_norm
    t7 = s_norm * temp_norm
    t8 = temp_norm * ph_norm
    t9 = s_norm * temp_norm * ph_norm
    t10 = temp_norm * temp_norm

    prediction = (
        params[0] * t1 +
        params[1] * t2 +
        params[2] * t3 +
        params[3] * t4 +
        params[4] * t5 +
        params[5] * t6 +
        params[6] * t7 +
        params[7] * t8 +
        params[8] * t9 +
        params[9] * t10
    )

    return prediction
# EVOLVE-BLOCK-END
