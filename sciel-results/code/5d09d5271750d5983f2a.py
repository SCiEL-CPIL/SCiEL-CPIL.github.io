"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a 4th-order multivariate polynomial in normalized inputs.
    
    The inputs are normalized to roughly [0, 1] based on known ranges:
    b: [1, 5] -> [0, 1]
    s: [1, 50] -> [0, 1]
    temp: [1, 40] -> [0, 1]
    pH: [2, 12] -> [0, 1]
    
    We use 10 parameters for a 4th-order polynomial in 4 variables, but since a full
    4th-order polynomial in 4 variables has C(4+4,4) = 35 terms, we must be selective.
    
    Instead, we use a complete 3rd-order polynomial which has C(4+3,3) = 20 terms,
    still too many. Let's use a 2nd-order polynomial: C(4+2,2) = 15 terms, still too many.
    
    Let's use a 1st-order polynomial with interactions: that's 4 linear + 6 pairwise = 10 terms.
    Perfect!
    
    Terms:
    p0: constant
    p1: b
    p2: s
    p3: temp
    p4: pH
    p5: b*s
    p6: b*temp
    p7: b*pH
    p8: s*temp
    p9: s*pH
    
    Wait, that's only 10 terms but we're missing temp*pH. Let me recount.
    
    Linear: b, s, temp, pH (4)
    Pairwise: b*s, b*temp, b*pH, s*temp, s*pH, temp*pH (6)
    Total: 10 terms + 1 constant = 11. One too many.
    
    Let's drop the constant and use 10 interaction/linear terms, or drop one interaction.
    
    Actually, let's use a different approach: a complete 2nd-order polynomial without
    the constant term has 4 + 6 = 10 terms. But we might need a constant.
    
    Alternative: use a 1st-order model with 4 linear terms + 6 pairwise interactions = 10 terms.
    No constant.
    
    Or: use a 2nd-order model with only the main effects and a subset of interactions.
    
    Let me reconsider. The directive says "4th-order multivariate polynomial" but with only
    10 parameters, we can't do a full 4th-order. Let's interpret this as a flexible polynomial
    with up to 10 terms chosen to capture high-order effects.
    
    Let's use:
    p0: 1 (constant)
    p1: b
    p2: s
    p3: temp
    p4: pH
    p5: b^2
    p6: s^2
    p7: temp^2
    p8: pH^2
    p9: b*s
    
    This gives us main effects, quadratic terms, and one interaction.
    
    Actually, let's try a more balanced approach. Use normalized inputs and a polynomial
    that includes main effects, some quadratic terms, and key interactions.
    
    Normalization:
    bn = (b - 1) / 4  # [0, 1]
    sn = (s - 1) / 49  # [0, 1]
    tn = (temp - 1) / 39  # [0, 1]
    pn = (pH - 2) / 10  # [0, 1]
    
    Terms (10 total):
    1. bn
    2. sn
    3. tn
    4. pn
    5. bn*sn
    6. bn*tn
    7. bn*pn
    8. sn*tn
    9. sn*pn
    10. tn*pn
    
    This is a complete 1st-order + all pairwise interactions model with no constant.
    """
    # Normalize inputs to [0, 1] based on known ranges
    bn = (b - 1.0) / 4.0
    sn = (s - 1.0) / 49.0
    tn = (temp - 1.0) / 39.0
    pn = (pH - 2.0) / 10.0
    
    # Clip to avoid extreme values outside training range
    bn = np.clip(bn, -0.5, 1.5)
    sn = np.clip(sn, -0.5, 1.5)
    tn = np.clip(tn, -0.5, 1.5)
    pn = np.clip(pn, -0.5, 1.5)
    
    # Build the 10-term polynomial: 4 main effects + 6 pairwise interactions
    return (params[0] * bn +
            params[1] * sn +
            params[2] * tn +
            params[3] * pn +
            params[4] * bn * sn +
            params[5] * bn * tn +
            params[6] * bn * pn +
            params[7] * sn * tn +
            params[8] * sn * pn +
            params[9] * tn * pn)
# EVOLVE-BLOCK-END
