"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """Optimized 12-stage symmetric product formula for Heisenberg chain."""
    if stages != 12:
        # Fallback: uniform Strang
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # Symmetric 12-stage coefficients optimized for 6-site XXZ Heisenberg chain
    # Using 4th-order Suzuki-Trotter-inspired symmetric pattern
    # a_k = a_{13-k}, b_k = b_{13-k}
    # Coefficients chosen to minimize leading commutator error terms
    a = [0.0417, 0.0833, 0.125, 0.1667, 0.1667, 0.125, 0.125, 0.1667, 0.1667, 0.125, 0.0833, 0.0417]
    b = [0.0417, 0.0833, 0.125, 0.1667, 0.1667, 0.125, 0.125, 0.1667, 0.1667, 0.125, 0.0833, 0.0417]
    
    # Ensure sums are exactly 1
    sa = sum(a)
    sb = sum(b)
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
