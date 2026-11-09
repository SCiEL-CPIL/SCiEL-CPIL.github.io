"""Rewritten program for the Černý problem (N=12).

Constructs the standard Černý automaton which is known to have a shortest
reset word of length (N-1)^2. For N=12, this length is 121.
"""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Letter 'a': cyclic shift i -> (i + 1) % N
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': 0 -> 0, i -> i - 1 for i > 0
    b = [0] + [i - 1 for i in range(1, N)]
    
    return [a, b]
# EVOLVE-BLOCK-END
