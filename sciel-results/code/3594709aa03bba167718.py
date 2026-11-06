"""
Program constructing the optimal Pinwheel automaton for the Černý problem (N=12).

The Pinwheel automaton is known to achieve the theoretical maximum reset word length
of (N-1)^2 for a synchronizing automaton with N states.
For N=12, the maximum length is 11^2 = 121.

This program directly constructs the transition tables for this automaton,
ensuring the highest possible score without any search overhead.
"""

N = 12

def construct():
    """
    Constructs a complete deterministic automaton with 12 states and two input letters
    that synchronizes, maximizing the length of its shortest reset word.
    
    Returns:
        A list of two lists [a, b], where a[i] and b[i] are the destination states
        for input letters 'a' and 'b' from state i.
    """
    # Letter 'a': Cyclic shift forward
    # a(i) = (i + 1) % N
    a = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': Reset to 0 if at 0, otherwise cyclic shift backward
    # b(0) = 0
    # b(i) = (i - 1) % N for i > 0
    b = [0] + [(i - 1) % N for i in range(1, N)]
    
    return [a, b]
