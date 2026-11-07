import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial Basis Expansion Model for E. coli growth rate.
    Uses a linear combination of main effects and selected interaction terms.
    This approach is distinct from mechanistic models and relies on flexible
    low-order polynomial approximation.
    
    Parameters:
    p0: Intercept
    p1: Coefficient for b
    p2: Coefficient for s
    p3: Coefficient for temp
    p4: Coefficient for pH
    p5: Coefficient for b*s
    p6: Coefficient for b*temp
    p7: Coefficient for s*temp
    p8: Coefficient for temp*pH
    p9: Coefficient for b*pH
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to improve numerical stability and condition number
    # b in [1, 5] -> [0, 1]
    b_norm = (b - 1.0) / 4.0
    # s in [1, 5] -> [0, 1]
    s_norm = (s - 1.0) / 4.0
    # temp in [1, 40] -> [0, 1]
    t_norm = (temp - 1.0) / 39.0
    # pH in [2, 12] -> [0, 1]
    ph_norm = (pH - 2.0) / 10.0
    
    # Construct basis functions
    # Main effects
    term_1 = b_norm
    term_2 = s_norm
    term_3 = t_norm
    term_4 = ph_norm
    
    # Interaction terms (selected to cover key pairwise relationships)
    term_5 = b_norm * s_norm
    term_6 = b_norm * t_norm
    term_7 = s_norm * t_norm
    term_8 = t_norm * ph_norm
    term_9 = b_norm * ph_norm
    
    # Linear combination
    pred = p0 + p1 * term_1 + p2 * term_2 + p3 * term_3 + p4 * term_4 + \
           p5 * term_5 + p6 * term_6 + p7 * term_7 + p8 * term_8 + p9 * term_9
    
    return pred
