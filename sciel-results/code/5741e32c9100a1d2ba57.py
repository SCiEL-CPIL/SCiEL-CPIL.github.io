"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def is_valid(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_random(rank, max_iters=10000):
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            U = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(rank)] for _ in range(p*n)]
            if is_valid(U, V, W, rank):
                return U, V, W
        return None
    
    # Try ranks from 16 upward
    for rank in range(16, best_rank):
        if time.time() > deadline:
            break
        result = try_random(rank, max_iters=5000)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
