import time

def _hamming_dist(a, b, bits=4):
    x = a ^ b
    d = 0
    for _ in range(bits):
        d += x & 1
        x >>= 1
    return d

def construct():
    n = 16
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            d = _hamming_dist(u, v, 4)
            if d == 1 or d == 4:
                edges.append([u, v])
    return edges
