"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    # The optimal lower bound k for the interval [k, n] such that
    # no a in [k, n] divides b+c for any a < b <= c in [k, n]
    # is determined by 3*k > 2*n.
    # k > 2*256 / 3 = 170.66...
    # So k = 171.
    k = 171
    return list(range(k, n + 1))
# EVOLVE-BLOCK-END
