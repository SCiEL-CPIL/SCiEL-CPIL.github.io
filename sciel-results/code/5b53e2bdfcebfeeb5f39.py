# EVOLVE-BLOCK-START
def construct():
    # The Černý automaton for N states has a reset threshold of (N-1)^2.
    # For N=12, this is 121. This is the known maximum for this size.
    # 'a' is the cycle 0->1->...->N-1->0
    # 'b' maps N-1 to 0 and fixes all other states.
    a = [(i + 1) % N for i in range(N)]
    b = list(range(N))
    b[N-1] = 0
    
    return [a, b]
# EVOLVE-BLOCK-END
