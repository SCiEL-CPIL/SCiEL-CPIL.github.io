def construct():
    """
    Return a polynomial with exactly 12 nonzero terms whose square has
    as few nonzero terms as possible.
    
    Uses P(x) = sum_{i=0}^{5} x^i - sum_{i=6}^{11} x^i
    This creates cancellations in P(x)^2 at specific positions.
    """
    # P(x) = 1 + x + x^2 + x^3 + x^4 + x^5 - x^6 - x^7 - x^8 - x^9 - x^10 - x^11
    poly = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    
    # Verify exactly 12 nonzero terms
    nonzero_count = sum(1 for c in poly if c != 0)
    assert nonzero_count == 12, f"Expected 12 nonzero terms, got {nonzero_count}"
    
    # Verify all entries are in [-64, 64]
    assert all(-64 <= c <= 64 for c in poly), "Coefficient out of range"
    
    # Verify length <= 41
    assert len(poly) <= 41, "Polynomial too long"
    
    # Verify nonzero final coefficient
    assert poly[-1] != 0, "Final coefficient must be nonzero"
    
    return poly
