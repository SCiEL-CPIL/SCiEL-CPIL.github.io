"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Černý (Pawlik) automaton for N=12.
    # Known to achieve the conjectured maximum reset threshold (N-1)^2 = 121.
    # Letter 0: cyclic permutation i -> (i+1) mod N
    # Letter 1: fixes 0, and for i>0 maps i -> i-1
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0 if i == 0 else i - 1 for i in range(N)]
    return [sigma0, sigma1]
# EVOLVE-BLOCK-END
