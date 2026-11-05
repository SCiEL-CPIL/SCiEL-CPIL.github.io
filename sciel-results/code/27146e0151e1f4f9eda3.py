"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def block_rank(si, sj, sk):
        """
        Return the minimum known rank for a block of size si x sj x sk.
        Uses recursive splitting: if any dimension is 1, rank is 1.
        For 2x2x2, rank is 7 (Strassen).
        For larger blocks, try splitting the largest dimension into 2+(d-2) or 1+(d-1).
        """
        if si <= 0 or sj <= 0 or sk <= 0:
            return 0
        # If any dimension is 1, the tensor is an outer product of vectors, rank 1
        if si == 1 or sj == 1 or sk == 1:
            return 1
        # If all are 2, use Strassen rank 7
        if si == 2 and sj == 2 and sk == 2:
            return 7
        # Schoolbook rank
        school = si * sj * sk
        best = school
        # Try splitting the largest dimension
        dims = [(si, 0), (sj, 1), (sk, 2)]
        dims.sort(key=lambda x: -x[0])
        for size, dim_idx in dims:
            if size <= 2:
                continue
            # Try splitting into 2 + (size-2)
            if size - 2 >= 1:
                parts = [size - 2, 2] if dim_idx != 0 else [size - 2, 2]
                # Reconstruct dimensions
                new_si = parts[0] if dim_idx == 0 else si
                new_sj = parts[0] if dim_idx == 1 else sj
                new_sk = parts[0] if dim_idx == 2 else sk
                old_si = parts[1] if dim_idx == 0 else si
                old_sj = parts[1] if dim_idx == 1 else sj
                old_sk = parts[1] if dim_idx == 2 else sk
                r1 = block_rank(new_si, new_sj, new_sk)
                r2 = block_rank(old_si, old_sj, old_sk)
                best = min(best, r1 + r2)
            # Try splitting into 1 + (size-1)
            if size - 1 >= 1:
                new_si = size - 1 if dim_idx == 0 else si
                new_sj = size - 1 if dim_idx == 1 else sj
                new_sk = size - 1 if dim_idx == 2 else sk
                old_si = 1 if dim_idx == 0 else si
                old_sj = 1 if dim_idx == 1 else sj
                old_sk = 1 if dim_idx == 2 else sk
                r1 = block_rank(new_si, new_sj, new_sk)
                r2 = block_rank(old_si, old_sj, old_sk)
                best = min(best, r1 + r2)
        return best

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_block(si, sj, sk, base_i, base_j, base_k, U, V, W, col):
        """
        Recursively build the decomposition for a block of size si x sj x sk
        at positions base_i, base_j, base_k in the global matrices.
        Returns the new column index.
        """
        if si <= 0 or sj <= 0 or sk <= 0:
            return col
        
        # If any dimension is 1, use rank-1 outer product
        if si == 1 or sj == 1 or sk == 1:
            U[base_i * m + base_j][col] = 1
            V[base_j * p + base_k][col] = 1
            W[base_k * n + base_i][col] = 1
            return col + 1
        
        # If all are 2, use Strassen
        if si == 2 and sj == 2 and sk == 2:
            strassen_terms = strassen_2x2x2_terms()
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = base_i + li, base_j + lj
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = base_j + lj, base_k + lk
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = base_k + lk, base_i + li
                    W[gk * n + gi][col] = val_w
                col += 1
            return col
        
        # For larger blocks, try splitting and use the best split
        # Find the best split by trying all possible ways to split each dimension
        dims = [(si, 0), (sj, 1), (sk, 2)]
        dims.sort(key=lambda x: -x[0])
        
        best_col = None
        best_split = None
        
        for size, dim_idx in dims:
            if size <= 2:
                continue
            # Try splitting into a + b where a+b = size
            for a in range(1, size):
                b = size - a
                # Compute rank for this split
                si1 = a if dim_idx == 0 else si
                sj1 = a if dim_idx == 1 else sj
                sk1 = a if dim_idx == 2 else sk
                si2 = b if dim_idx == 0 else si
                sj2 = b if dim_idx == 1 else sj
                sk2 = b if dim_idx == 2 else sk
                r = block_rank(si1, sj1, sk1) + block_rank(si2, sj2, sk2)
                if best_col is None or r < best_col:
                    best_col = r
                    best_split = (dim_idx, a, b)
        
        if best_split is None:
            # Fallback to schoolbook
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = base_i + li, base_j + lj, base_k + lk
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1
            return col
        
        dim_idx, a, b = best_split
        # Build first part
        if dim_idx == 0:
            col = build_block(a, sj, sk, base_i, base_j, base_k, U, V, W, col)
            col = build_block(b, sj, sk, base_i + a, base_j, base_k, U, V, W, col)
        elif dim_idx == 1:
            col = build_block(si, a, sk, base_i, base_j, base_k, U, V, W, col)
            col = build_block(si, b, sk, base_i, base_j + a, base_k, U, V, W, col)
        else:
            col = build_block(si, sj, a, base_i, base_j, base_k, U, V, W, col)
            col = build_block(si, sj, b, base_i, base_j, base_k + a, U, V, W, col)
        
        return col

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try the recursive approach with full n x m x p block
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = build_block(n, m, p, 0, 0, 0, U, V, W, 0)
    
    # Trim to actual rank used
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    result = {"U": U, "V": V, "W": W}
    
    if col < best_rank and verify(U, V, W, col):
        best_result = result
        best_rank = col

    return best_result
# EVOLVE-BLOCK-END
