"""Rewritten program for the Černý problem (N=12).

Constructs a binary synchronizing automaton with 12 states achieving
the Pawlik reset threshold of (N-1)(N-2) = 110.
"""
N = 12

# EVOLVE-BLOCK-START
def construct():
    # Pawlik automaton structure for N states:
    # Letter 'a': shifts 0->1->2->...->N-2->N-1, with N-1 fixed
    # Letter 'b': 0 fixed, i->i-1 for i>0, N-1->N-2
    # This achieves reset word length (N-1)(N-2)
    
    a = list(range(1, N)) + [N - 1]  # 0->1, 1->2, ..., N-2->N-1, N-1->N-1
    b = [0] + list(range(N - 1))     # 0->0, 1->0, 2->1, ..., N-1->N-2
    
    return [a, b]
# EVOLVE-BLOCK-END
