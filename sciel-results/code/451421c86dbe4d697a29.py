N = 12

def construct():
    # Symbol 'a': cyclic shift
    a = [(i + 1) % N for i in range(N)]
    
    # Symbol 'b': 0->0, N-1->0, others fixed
    b = [0] * N
    b[0] = 0
    b[N - 1] = 0
    for i in range(1, N - 1):
        b[i] = i
        
    return [a, b]
