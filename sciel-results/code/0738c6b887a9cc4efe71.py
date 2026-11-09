import time
import random
from fractions import Fraction
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 5, 6
    rank_schoolbook = n * m * p  # 60
    
    def schoolbook():
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != Fraction(1):
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j1 == j2 and k1 == k2 and i1 == i2:
                        continue
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != Fraction(0):
                        return False
        return True
    
    # Return schoolbook decomposition
    return schoolbook()
# EVOLVE-BLOCK-END
