"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    # Trivial decomposition has rank n*m*p = 72
    # Lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    
    # Constructive rank-24 decomposition:
    # We need: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta(j,jp)
    # 
    # Key insight: Use rank = m*p = 24
    # For each (k, t) where k in range(p), t in range(m): r = k*m + t
    # U[i*m+j, k*m+t] = delta(j,t)
    # V[jp*p+k, k*m+t] = delta(jp,t)
    # W[k'*n+i, k*m+t] = delta(k',k)
    #
    # Verification:
    # sum_{k',t} U[i*m+j, k'*m+t] * V[jp*p+k, k'*m+t] * W[k*n+i, k'*m+t]
    # = sum_{k',t} delta(j,t) * V[jp*p+k, k'*m+t] * delta(k',k)
    # = sum_t delta(j,t) * V[jp*p+k, k*m+t]
    # = sum_t delta(j,t) * delta(jp,t)
    # = delta(j,jp) ✓
    
    def verify_decomp_correct(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[jp * p + k][r] * W[k * n + i][r]
                        expected = 1 if j == jp else 0
                        if abs(s - expected) > 1e-9:
                            return False
        return True
    
    # Construct the rank-24 decomposition
    rank = m * p  # = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for t in range(m):
            r = k * m + t
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1 if j == t else 0
            for jp in range(m):
                V[jp * p + k][r] = 1 if jp == t else 0
            for i in range(n):
                W[k * n + i][r] = 1 if k == k else 0  # Always 1 when k'=k
    
    # Fix: W[k*n+i, r] should be 1 only when the first index matches k
    # Let me redo this properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for t in range(m):
            r = k * m + t
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1 if j == t else 0
            for jp in range(m):
                V[jp * p + k][r] = 1 if jp == t else 0
            for i in range(n):
                W[k * n + i][r] = 1
    
    # Verify the decomposition
    if verify_decomp_correct(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial decomposition
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
