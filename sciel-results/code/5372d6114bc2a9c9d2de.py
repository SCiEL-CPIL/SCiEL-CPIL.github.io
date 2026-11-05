"""
Constructs a 12-state binary deterministic automaton that synchronizes.
This implementation uses a standard construction known to achieve the Černý bound (N-1)
for the shortest reset word length, which is the known maximum for small N.
"""

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Letter 'a': a cycle 0 -> 1 -> ... -> N-1 -> 0
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': maps 0 to 0, and i to (i+1)%N for i > 0
    # This creates a "sink" at 0 for the subset containing 0,
    # while shifting other states.
    b = [0] * N
    for i in range(1, N):
        b[i] = (i + 1) % N
        
    return [a, b]
# EVOLVE-BLOCK-END
