def construct():
    h = [0.0] * N
    h[N // 2] = 1e-6
    return h
