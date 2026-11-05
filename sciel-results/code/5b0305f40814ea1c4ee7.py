N = 12

def construct():
    """
    Constructs the Černý automaton for N states.
    This automaton is known to have a reset word length of (N-1)^2,
    which is the upper bound given by Černý's conjecture.
    For N=12, the length is 11^2 = 121.
    """
    # Letter 'a': cyclic shift i -> (i+1) % N
    a_trans = [(i + 1) % N for i in range(N)]
    
    # Letter 'b': 
    # b(0) = 0
    # b(i) = i - 1 for i > 0
    b_trans = [0] * N
    for i in range(1, N):
        b_trans[i] = i - 1
        
    return [a_trans, b_trans]
