"""Rewritten program for 3x5x6 matrix multiplication rank search."""
DIMENSIONS = [3, 5, 6]

import sys
import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid matrix multiplication decomposition."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_pattern, V_pattern, W_pattern) where patterns are
    dicts mapping (local_i, local_j) -> value for U, etc.
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2x2: i,j,k in {0,1}, so U indices are i*2+j, V indices are j*2+k, W indices are k*2+i
    """
    terms = []
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # Contributes to C00 (i=0,k=0) and C11 (i=1,k=1)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # Term 2: M2 = (A10+A11)B00
    # Contributes to C10 (i=1,k=0) and C11 (i=1,k=1)
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(0,1): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # Term 3: M3 = A00(B01-B11)
    # Contributes to C01 (i=0,k=1) and C11 (i=1,k=1)
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(1,0): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # Term 4: M4 = A11(B10-B00)
    # Contributes to C00 (i=0,k=0) and C10 (i=1,k=0)
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (0,1): 1}
    terms.append((u4, v4, w4))
    
    # Term 5: M5 = (A00+A01)B11
    # Contributes to C00 (i=0,k=0) and C01 (i=0,k=1)
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (1,0): 1}
    terms.append((u5, v5, w5))
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # Contributes to C11 (i=1,k=1)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # Contributes to C00 (i=0,k=0)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def _build_block_strassen_decomposition(n, m, p):
    """
    Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
    
    Partition:
    - i (rows of A, 0..n-1): group into pairs + remainder
    - j (cols of A, 0..m-1): group into pairs + remainder
    - k (cols of B, 0..p-1): group into pairs + remainder
    
    For each sub-block (Ig, Jg, Kg):
    - If all sizes are 2: use Strassen (rank 7)
    - Otherwise: use schoolbook for that sub-block
    """
    # Define groups: pair up elements, leave remainder as singleton
    def make_groups(size):
        groups = []
        i = 0
        while i < size:
            if i + 1 < size:
                groups.append([i, i+1])
                i += 2
            else:
                groups.append([i])
                i += 1
        return groups
    
    i_groups = make_groups(n)
    j_groups = make_groups(m)
    k_groups = make_groups(p)
    
    # First pass: count total rank needed
    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                if si == 2 and sj == 2 and sk == 2:
                    r = 7  # Strassen
                else:
                    r = si * sj * sk  # schoolbook
                total_rank += r
                block_info.append((ig, jg, kg, r))
    
    # Allocate columns
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    strassen_terms = _strassen_2x2x2_terms()
    
    for ig, jg, kg, r in block_info:
        si, sj, sk = len(ig), len(jg), len(kg)
        if si == 2 and sj == 2 and sk == 2:
            # Use Strassen
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            # Schoolbook for this sub-block
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}

def _construct():
    """Main construction function."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block Strassen decomposition
    if time.time() < deadline - 10:
        result = _build_block_strassen_decomposition(n, m, p)
        rank = result.pop("rank")
        
        # Verify
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, rank, tol=1e-9):
            if rank < best_rank:
                best = result
                best_rank = rank
    
    return best

def construct():
    return _construct()

# EVOLVE-BLOCK-END
