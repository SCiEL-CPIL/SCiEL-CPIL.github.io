"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import fractions
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strategy: Use structured algebraic decomposition based on Strassen-like ideas.
    # For 2x4x6, we can view it as 2x2x6 blocks.
    # Rank(2x2x6) is likely 24 (schoolbook) or maybe less?
    # Actually, for 2x2xn, the rank is 2n if n is even?
    # Let's try to find a decomposition for 2x2x6 first.
    # Then combine two such blocks for 2x4x6?
    # 2x4x6 is (2x2x6) concatenated with (2x2x6) in the m dimension.
    # If Rank(2x2x6) = R, then Rank(2x4x6) <= 2*R.
    # If R=24, then Rank <= 48.
    # Is Rank(2x2x6) < 24?
    # Known: Rank(2x2x2)=7.
    # Rank(2x2x3)=12? (Schoolbook is 12).
    # Rank(2x2x4)=16?
    # Rank(2x2x6)=24?
    # It is known that for 2x2xn, the rank is 2n for n>=4?
    # Actually, for 2x2x2, it's 7 < 8.
    # For 2x2x3, it's 12 = 2*3.
    # For 2x2x4, it's 16 = 2*4.
    # So likely Rank(2x2x6) = 24.
    # Thus Rank(2x4x6) <= 48.
    
    # Can we do better by mixing the blocks?
    # The tensor is T = T1 + T2 where T1 is the first 2x2x6 block and T2 is the second.
    # T1 and T2 are "disjoint" in the index space?
    # T1 involves j=0,1. T2 involves j=2,3.
    # The indices a, b, c are coupled.
    # a = i*m + j.
    # For T1: a in {0,1,2,3} (i=0,1; j=0,1).
    # For T2: a in {4,5,6,7} (i=0,1; j=2,3).
    # b = j*p + k.
    # For T1: b in {0..11} (j=0,1).
    # For T2: b in {12..23} (j=2,3).
    # c = k*n + i.
    # c is the same for both.
    # So the tensors share the c-index.
    # This means we cannot simply add the ranks.
    # We might be able to share some terms in W.
    
    # Let's try a random search for rank 47, 46, 45, 44.
    # But with a smarter initialization: start from schoolbook and perturb.
    
    if time.time() < deadline - 20:
        for r in [47, 46, 45, 44]:
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                # Initialize with schoolbook and drop one column?
                # Or just random?
                # Random is better for exploring the space.
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
                if verify(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    return best_result
    
    return best_result

# EVOLVE-BLOCK-END
