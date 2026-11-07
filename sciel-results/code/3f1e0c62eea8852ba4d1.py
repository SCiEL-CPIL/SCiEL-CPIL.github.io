"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a 4th-order multivariate polynomial in normalized inputs.
    Normalization is fixed based on training ranges to keep the model well-conditioned.
    """
    # Fixed normalization constants based on training ranges
    # b: [1, 4.998] -> center ~3, scale ~2
    # s: [1, 4.999] -> center ~3, scale ~2
    # temp: [1.002, 39.98] -> center ~20.5, scale ~20
    # pH: [2, 12] -> center ~7, scale ~5
    
    # Normalize inputs to roughly [-1, 1]
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    temp_n = (temp - 20.5) / 20.0
    ph_n = (pH - 7.0) / 5.0
    
    # Extract params
    p = params
    
    # Construct 4th-order multivariate polynomial
    # We have 10 parameters to allocate.
    # Let's use a structured set of terms.
    # Terms:
    # 1. Constant: p[0]
    # 2. Linear: p[1]*b_n + p[2]*s_n + p[3]*temp_n + p[4]*ph_n
    # 3. Quadratic interactions and main effects:
    #    p[5]*(b_n**2 + s_n**2 + temp_n**2 + ph_n**2)  (isotropic quadratic)
    #    p[6]*(b_n*s_n + b_n*temp_n + b_n*ph_n + s_n*temp_n + s_n*ph_n + temp_n*ph_n) (isotropic quadratic interactions)
    # 4. Cubic terms:
    #    p[7]*(b_n**3 + s_n**3 + temp_n**3 + ph_n**3) (isotropic cubic)
    # 5. Quartic terms:
    #    p[8]*(b_n**4 + s_n**4 + temp_n**4 + ph_n**4) (isotropic quartic)
    # 6. High-order interaction / modulation:
    #    p[9]*(b_n**2 * s_n**2 + b_n**2 * temp_n**2 + ... )? 
    #    Let's try a specific 4th order interaction term that might capture saturation or inhibition.
    #    Or perhaps a term like (b_n*s_n*temp_n*ph_n) * p[9]?
    
    # Let's refine the term selection to be more flexible while staying within 10 params.
    # Maybe separate some of the isotropic terms?
    
    # Revised plan:
    # p[0]: Intercept
    # p[1]: b_n
    # p[2]: s_n
    # p[3]: temp_n
    # p[4]: ph_n
    # p[5]: b_n**2
    # p[6]: s_n**2
    # p[7]: temp_n**2
    # p[8]: ph_n**2
    # p[9]: b_n*s_n (strongest interaction?)
    
    # This uses all 10 params for a specific subset of terms.
    # Is this better than isotropic? The optimizer can fit different curvatures for each variable.
    # This is a standard polynomial regression approach.
    
    term_0 = p[0]
    term_1 = p[1] * b_n
    term_2 = p[2] * s_n
    term_3 = p[3] * temp_n
    term_4 = p[4] * ph_n
    term_5 = p[5] * (b_n ** 2)
    term_6 = p[6] * (s_n ** 2)
    term_7 = p[7] * (temp_n ** 2)
    term_8 = p[8] * (ph_n ** 2)
    term_9 = p[9] * (b_n * s_n)
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
# EVOLVE-BLOCK-END
