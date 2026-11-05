N = 12

# EVOLVE-BLOCK-START
def construct():
    # Černý automaton construction
    # a: 0 -> 0, i -> i-1 for i > 0
    # b: i -> i+1 for i < N-1, N-1 -> N-1
    a = [0] * N
    b = [0] * N
    
    for i in range(N):
        if i == 0:
            a[i] = 0
            b[i] = 1
        else:
            a[i] = i - 1
            if i < N - 1:
                b[i] = i + 1
            else:
                b[i] = i
                
    return [a, b]
# EVOLVE-BLOCK-END
