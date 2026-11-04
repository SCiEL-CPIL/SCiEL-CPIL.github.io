"""
Strategy: 
1. Use 4 blocks of 2x3x2 decomposition.
2. Each 2x3x2 block has rank 10 (Strassen 2x2x2 rank 7 + 3 additional terms).
3. Total rank 40.
4. Fallback to rank 44 (Strassen + Schoolbook) if rank 10 block construction fails.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    S_W = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]

    def _validate_block(U_b, V_b, W_b, rank):
        for i in range(2):
            for a in range(3):
                for k in range(2):
                    acc = 0
                    for r in range(rank):
                        acc += U_b[i*3+a][r] * V_b[a*2+k][r] * W_b[k*2+i][r]
                    if acc != 1:
                        return False
        return True

    def _search_block_decomp(target_rank):
        import random
        random.seed(42)
        max_restarts = 5000
        for _ in range(max_restarts):
            if not _elapsed_ok():
                return None
            U_b = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(6)]
            V_b = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(6)]
            W_b = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(4)]
            if _validate_block(U_b, V_b, W_b, target_rank):
                return U_b, V_b, W_b
        return None

    def _try_block_tiling():
        # Try ranks from 6 to 10
        for r in range(6, 11):
            if not _elapsed_ok():
                return None
            block = _search_block_decomp(r)
            if block:
                U_b, V_b, W_b = block
                rank = 4 * r
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                for b in range(4):
                    k0 = b * 2
                    for rr in range(r):
                        col = b * r + rr
                        for i in range(n):
                            for a in range(m):
                                U[i*m+a][col] = U_b[i*3+a][rr]
                        for a in range(m):
                            for k in range(2):
                                V[a*p + (k0+k)][col] = V_b[a*2+k][rr]
                        for k in range(2):
                            for i in range(n):
                                W[(k0+k)*n + i][col] = W_b[k*2+i][rr]
                                
                candidate = {"U": U, "V": V, "W": W}
                if _validate(candidate):
                    return candidate
        return None

    block_candidate = _try_block_tiling()
    if block_candidate:
        return block_candidate

    # Fallback to Rank 44
    rank_base = 28
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U_base[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V_base[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W_base[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank_base):
        for i in range(n):
            for a in range(2):
                U[i*m+a][r] = U_base[i*m+a][r]
        for a in range(2):
            for k in range(p):
                V[a*p+k][r] = V_base[a*p+k][r]
        for k in range(p):
            for i in range(n):
                W[k*n+i][r] = W_base[k*n+i][r]
                
    col = rank_base
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
