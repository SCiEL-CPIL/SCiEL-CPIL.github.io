"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 12-state binary synchronizing automaton based on the Pawlik construction.
    This automaton is known to have a high shortest reset word length, specifically (N-1)*(N-2).
    For N=12, the expected threshold is 110.
    
    Structure:
    Letter 'a' (index 0): Cyclic shift i -> (i+1) % N
    Letter 'b' (index 1): 
        - Maps 0 to 0
        - Maps i to i-1 for 1 <= i <= N//2
        - Maps i to i for N//2 < i < N
    """
    m = N // 2
    
    # Letter 'a': Cycle
    trans_a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': Pawlik-style mapping
    trans_b = [0] * N
    trans_b[0] = 0
    for i in range(1, m + 1):
        trans_b[i] = i - 1
    for i in range(m + 1, N):
        trans_b[i] = i
        
    return [trans_a, trans_b]
# EVOLVE-BLOCK-END
