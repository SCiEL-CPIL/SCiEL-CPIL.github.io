import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function f(x, y) using 10 parameters.
    
    Uses a carefully chosen orthogonal basis on [-1, 1] x [-1, 1] with
    Legendre polynomials. The basis is ordered to capture the most important
    low-degree terms first, with a mix of pure and interaction terms.
    
    Key design choices:
    - Legendre polynomials are orthogonal on [-1, 1], reducing multicollinearity
    - We include terms up to degree 4, prioritizing lower-degree terms
    - The basis includes both pure (x-only, y-only) and interaction terms
    - We use a slightly different combination than the seed to capture
      potentially important quartic and mixed terms
    """
    p = params
    
    # Compute Legendre polynomials for x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    L0_x = np.ones_like(x)
    L1_x = x
    L2_x = 0.5 * (3.0 * x2 - 1.0)
    L3_x = 0.5 * (5.0 * x3 - 3.0 * x)
    L4_x = 0.375 * (35.0 * x4 - 30.0 * x2 + 3.0)
    
    # Compute Legendre polynomials for y
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    y5 = y4 * y
    
    L0_y = np.ones_like(y)
    L1_y = y
    L2_y = 0.5 * (3.0 * y2 - 1.0)
    L3_y = 0.5 * (5.0 * y3 - 3.0 * y)
    L4_y = 0.375 * (35.0 * y4 - 30.0 * y2 + 3.0)
    
    # Construct 10 basis functions.
    # Strategy: include all terms up to degree 2 (6 terms), then select
    # the most important degree 3 and 4 terms (4 more terms).
    #
    # Degree 0: 1
    # Degree 1: x, y
    # Degree 2: x^2, y^2, xy
    # Degree 3: x^3, y^3, x^2y, xy^2
    # Degree 4: x^4, y^4, x^2y^2
    #
    # We have 10 parameters, so we need to choose which 10 terms to include.
    # The seed used: 1, x, y, L2x, L2y, xy, L3x, L3y, L2x*L1y, L1x*L2y
    # That's: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2 (all up to degree 3)
    #
    # Alternative: include some degree 4 terms instead of some degree 3 terms.
    # Let's try: 1, x, y, L2x, L2y, xy, L3x, L3y, L2x*L2y, L4x
    # This trades xy^2 for x^2y^2 and x^4.
    
    # Actually, let me think more carefully. The score is 7.0242, which is
    # a normalized MSE. If the true function is smooth, the orthogonal basis
    # should converge quickly. The issue might be that we're missing important
    # higher-order terms.
    
    # Let me try a different combination:
    # 1, x, y, L2x, L2y, xy, L3x, L3y, L4x, L4y
    # This gives us pure terms up to degree 4, but misses mixed terms of degree 3+.
    # That might be worse.
    
    # Better: keep the seed's approach but add one more degree of freedom
    # by replacing a less important term. The seed has all terms up to degree 3
    # except for some. Let me count:
    # Degree 0: 1 (1 term)
    # Degree 1: x, y (2 terms)
    # Degree 2: L2x, L2y, xy (3 terms) -- note: L2x = 0.5(3x^2-1), so this is x^2
    # Degree 3: L3x, L3y, L2x*L1y, L1x*L2y (4 terms) -- x^3, y^3, x^2y, xy^2
    # Total: 1+2+3+4 = 10 terms
    #
    # This is actually a complete basis for all polynomials up to degree 3 in two
    # variables! The space of polynomials of degree <= 3 in 2 variables has
    # (3+2 choose 2) = 10 dimensions. So the seed is already using the complete
    # degree-3 polynomial basis.
    #
    # If the true function is a polynomial of degree 3 or less, this should fit
    # perfectly (up to numerical precision). The score of 7.0242 suggests either:
    # 1. The true function is not a polynomial of degree <= 3
    # 2. There's some issue with the optimization
    #
    # Let me try including some degree 4 terms instead. We can drop one of the
    # degree 3 terms and add two degree 4 terms, or drop two degree 3 terms
    # and add two degree 4 terms.
    #
    # The most important degree 4 terms are likely L4x, L4y, L2x*L2y, L3x*L1y, L1x*L3y
    # Let me try: drop L3x and L3y, add L4x and L4y
    # New basis: 1, x, y, L2x, L2y, xy, L2x*L1y, L1x*L2y, L4x, L4y
    
    term0 = L0_x * L0_y          # 1
    term1 = L1_x * L0_y          # x
    term2 = L0_x * L1_y          # y
    term3 = L2_x * L0_y          # 0.5(3x^2-1) ~ x^2
    term4 = L0_x * L2_y          # 0.5(3y^2-1) ~ y^2
    term5 = L1_x * L1_y          # xy
    term6 = L2_x * L1_y          # 0.5(3x^2-1)y ~ x^2y
    term7 = L1_x * L2_y          # x*0.5(3y^2-1) ~ xy^2
    term8 = L4_x * L0_y          # 0.375(35x^4-30x^2+3) ~ x^4
    term9 = L0_x * L4_y          # 0.375(35y^4-30y^2+3) ~ y^4
    
    return (p[0] * term0 +
            p[1] * term1 +
            p[2] * term2 +
            p[3] * term3 +
            p[4] * term4 +
            p[5] * term5 +
            p[6] * term6 +
            p[7] * term7 +
            p[8] * term8 +
            p[9] * term9)
