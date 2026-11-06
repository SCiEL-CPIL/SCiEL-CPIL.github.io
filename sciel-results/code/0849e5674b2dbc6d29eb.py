"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a synchronizing automaton with 12 states and 2 letters.
    Uses a cycle for letter 0 and a decrement-with-fixed-point for letter 1.
    This construction achieves a reset threshold of N-1 = 11.
    """
    # Letter 0: cycle i -> (i+1) mod N
    trans_0 = [(i + 1) % N for i in range(N)]
    
    # Letter 1: 0 -> 0, i -> i-1 for i > 0
    trans_1 = [0] * N
    for i in range(1, N):
        trans_1[i] = i - 1
    
    return [trans_0, trans_1]
# EVOLVE-BLOCK-END
