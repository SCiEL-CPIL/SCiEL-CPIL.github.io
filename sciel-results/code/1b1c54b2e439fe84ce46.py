"""Improved Strassen-based matrix multiplication tensor decomposition for 3x5x6."""

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 6

    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook

    # Block partitions for Strassen optimization
    # i_blocks: partition of {0,1,2}
    # j_blocks: partition of {0,1,2,3,4}
    # k_blocks: partition of {0,1,2,3,4,5}
    
    i_blocks = [
        [0, 1],
        [2]
    ]
    j_blocks = [
        [0, 1],
        [2, 3],
        [4]
    ]
    k_blocks = [
        [0, 1],
        [2, 3],
        [4, 5]
    ]

    def strassen_222_terms(i0, i1, j0, j1, k0, k1):
        """Return 7 rank-1 tensor terms for 2x2x2 Strassen decomposition."""
        terms = []
        # Each term is a tuple (u_dict, v_dict, w_dict) where dicts map indices to coefficients
        
        # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        u1 = {(i0,j0): 1, (i1,j1): 1}
        v1 = {(j0,k0): 1, (j1,k1): 1}
        w1 = {(k0,i0): 1, (k1,i1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        u2 = {(i1,j0): 1, (i1,j1): 1}
        v2 = {(j0,k0): 1}
        w2 = {(k0,i1): 1, (k1,i1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: A00(B10-B11) -> C00 += M3
        u3 = {(i0,j0): 1}
        v3 = {(j1,k0): 1, (j1,k1): -1}
        w3 = {(k0,i0): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: A11(B01-B00) -> C10 += M4
        u4 = {(i1,j1): 1}
        v4 = {(j0,k1): 1, (j0,k0): -1}
        w4 = {(k1,i1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: (A00+A01)B11 -> C10 += M5, C00 -= M5
        u5 = {(i0,j0): 1, (i0,j1): 1}
        v5 = {(j1,k1): 1}
        w5 = {(k1,i1): 1, (k0,i0): -1}
        terms.append((u5, v5, w5))
        
        # Term 6: (A10-A00)(B00+B01) -> C11 += M6
        u6 = {(i1,j0): 1, (i0,j0): -1}
        v6 = {(j0,k0): 1, (j0,k1): 1}
        w6 = {(k1,i1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: (A01-A11)(B10+B11) -> C01 += M7, C10 -= M7
        u7 = {(i0,j1): 1, (i1,j1): -1}
        v7 = {(j1,k0): 1, (j1,k1): 1}
        w7 = {(k0,i0): 1, (k1,i1): -1}
        terms.append((u7, v7, w7))
        
        return terms

    def schoolbook_terms(i_list, j_list, k_list):
        """Return rank-1 tensor terms for schoolbook decomposition of a block."""
        terms = []
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
        return terms

    # Accumulate all terms
    all_terms = []
    
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                i_size = len(i_blk)
                j_size = len(j_blk)
                k_size = len(k_blk)
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    i0, i1 = i_blk
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    terms = strassen_222_terms(i0, i1, j0, j1, k0, k1)
                    all_terms.extend(terms)
                else:
                    terms = schoolbook_terms(i_blk, j_blk, k_blk)
                    all_terms.extend(terms)

    # Build U, V, W matrices
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val

    # Verify correctness
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if is_valid and rank < best_rank:
        return {"U": U, "V": V, "W": W}
    else:
        return best_result
# EVOLVE-BLOCK-END
