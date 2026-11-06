"""Physics: product-formula design for Heisenberg-chain time evolution."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    Symmetric product formula with optimized coefficients.
    """
    # Use a symmetric formula with 12 stages.
    # Based on the 4th-order Suzuki formula, adapted.
    
    # c = 1/(2 - 2^(1/3))
    c = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    
    # 4th-order formula A coefficients: [a, b, b, a] where a + 2b = 1
    # 4th-order formula B coefficients: [c, d, c] where 2c + d = 1
    
    # For the 4th-order formula, the correct coefficients are:
    # a = 1/(2(2 - 2^(1/3))), b = 1 - 2a
    # c = 1/(2(2 - 2^(1/3))), d = 1 - 2c
    
    a_coeff = 1.0 / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0)))
    b_coeff = 1.0 - 2.0 * a_coeff
    
    # Map to 12 stages with symmetry:
    # a = [a_coeff, 0, 0, b_coeff, 0, 0, b_coeff, 0, 0, a_coeff, 0, 0]
    # b = [a_coeff, 0, 0, 0, 0, b_coeff, 0, 0, 0, 0, 0, a_coeff]
    
    a = [a_coeff, 0.0, 0.0, b_coeff, 0.0, 0.0, b_coeff, 0.0, 0.0, a_coeff, 0.0, 0.0]
    b = [a_coeff, 0.0, 0.0, 0.0, 0.0, b_coeff, 0.0, 0.0, 0.0, 0.0, 0.0, a_coeff]
    
    # Check sums
    s_a = sum(a)
    s_b = sum(b)
    
    # Scale to ensure sum = 1
    if abs(s_a) > 1e-15:
        a = [x / s_a for x in a]
    if abs(s_b) > 1e-15:
        b = [x / s_b for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
