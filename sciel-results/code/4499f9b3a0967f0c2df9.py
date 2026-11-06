N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the Pinwheel automaton for N=12, which achieves the 
    theoretical maximum reset length of (N-1)^2 = 121.
    
    a: 0->1, 1->2, ..., N-2->N-1, N-1->N-1
    b: 0->0, 1->0, 2->1, ..., N-1->N-2
    """
    a = [0] * N
    b = [0] * N
    
    # Define transitions for 'a'
    a[0] = 1
    for i in range(1, N - 1):
        a[i] = i + 1
    a[N - 1] = N - 1
    
    # Define transitions for 'b'
    b[0] = 0
    b[1] = 0
    for i in range(2, N):
        b[i] = i - 1
        
    return [a, b]
# EVOLVE-BLOCK-END
