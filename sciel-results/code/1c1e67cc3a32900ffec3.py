def construct():
    N = 600
    h = [0.0] * N
    for i in range(0, N, 2):
        h[i] = 1.0
    return h
