import time
import random
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify that U @ diag(1) @ V^T == W (with correct trace convention).
    Rank r = len(U[0]) = len(V[0]) = len(W[0])
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    """
    r = len(U[0])
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n*m)])
        vc = np.array([V[j][c] for j in range(m*p)])
        wc = np.array([W[k][c] for k in range(p*n)])
        # Reshape to matrices
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat  # (n, p)
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col):
    """Apply Strassen to 2x2 block at row i0..i0+1, col j0..j0+1, producing k0..k0+1.
    Returns new column index."""
    # The 7 multiplications:
    a_coeffs = [
        [(0,1), (1,1)],   # M1: A00, A11
        [(1,0), (1,1)],   # M2: A10, A11
        [(0,0)],          # M3: A00
        [(1,1)],          # M4: A11
        [(0,0), (0,1)],   # M5: A00, A01
        [(1,0), (0,0)],   # M6: A10, A00 (with sign: A10 - A00)
        [(0,1), (1,1)],   # M7: A01, A11 (with sign: A01 - A11)
    ]
    a_signs = [
        [1, 1],
        [1, 1],
        [1],
        [1],
        [1, 1],
        [1, -1],
        [1, -1],
    ]
    
    b_coeffs = [
        [(0,0), (1,1)],   # M1: B00, B11
        [(0,0)],          # M2: B00
        [(1,0), (1,1)],   # M3: B10, B11 (sign: B10 - B11)
        [(0,1), (0,0)],   # M4: B01, B00 (sign: B01 - B00)
        [(1,1)],          # M5: B11
        [(0,0), (0,1)],   # M6: B00, B01
        [(1,0), (1,1)],   # M7: B10, B11
    ]
    b_signs = [
        [1, 1],
        [1],
        [1, -1],
        [1, -1],
        [1],
        [1, 1],
        [1, 1],
    ]
    
    c_contribs = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): -1},     # M2
        {(0,1): 1, (1,1): 1},      # M3
        {(0,0): 1, (1,0): 1},      # M4
        {(0,0): -1, (0,1): 1},     # M5
        {(1,1): 1},                # M6
        {(0,0): 1},                # M7
    ]
    
    for t in range(7):
        c = col + t
        # U contributions
        for (di, dj), coeff in zip(a_coeffs[t], a_signs[t]):
            row = (i0 + di) * m + (j0 + dj)
            U[row][c] += coeff
        # V contributions
        for (dj, dk), coeff in zip(b_coeffs[t], b_signs[t]):
            row = (j0 + dj) * p + (k0 + dk)
            V[row][c] += coeff
        # W contributions
        for (dk, di), coeff in c_contribs[t].items():
            row = (k0 + dk) * n + (i0 + di)
            W[row][c] += coeff
    
    return col + 7

def _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col):
    """Schoolbook for sub-block of size hi x hj x hk."""
    for di in range(hi):
        for dj in range(hj):
            for dk in range(hk):
                U[(i0 + di) * m + (j0 + dj)][col] += 1
                V[(j0 + dj) * p + (k0 + dk)][col] += 1
                W[(k0 + dk) * n + (i0 + di)][col] += 1
                col += 1
    return col

def _construct_hybrid(splits_i, splits_j, splits_k, use_strassen_for_2x2):
    """Construct using block decomposition."""
    n, m, p = DIMENSIONS
    # Estimate max rank
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    i_offsets = [0]
    for s in splits_i[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in splits_j[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in splits_k[:-1]:
        k_offsets.append(k_offsets[-1] + s)
    
    for ii in range(len(splits_i)):
        for jj in range(len(splits_j)):
            for kk in range(len(splits_k)):
                hi = splits_i[ii]
                hj = splits_j[jj]
                hk = splits_k[kk]
                i0 = i_offsets[ii]
                j0 = j_offsets[jj]
                k0 = k_offsets[kk]
                
                if hi == 2 and hj == 2 and hk == 2 and use_strassen_for_2x2:
                    col = _construct_strassen_block(i0, j0, k0, n, m, p, U, V, W, col)
                else:
                    col = _construct_schoolbook_block(i0, j0, k0, hi, hj, hk, U, V, W, col)
    
    # Trim trailing zero columns
    last_col = 0
    for c in range(col - 1, -1, -1):
        nz = False
        for row in range(n * m):
            if U[row][c] != 0:
                nz = True
                break
        if not nz:
            for row in range(m * p):
                if V[row][c] != 0:
                    nz = True
                    break
        if not nz:
            for row in range(p * n):
                if W[row][c] != 0:
                    nz = True
                    break
        if nz:
            last_col = c
            break
    final_rank = last_col + 1
    
    U_trim = [row[:final_rank] for row in U]
    V_trim = [row[:final_rank] for row in V]
    W_trim = [row[:final_rank] for row in W]
    
    return {"U": U_trim, "V": V_trim, "W": W_trim, "_rank": final_rank, "_col": col}

def _all_compositions(total, min_part=1, max_parts=4):
    """Generate all compositions of total into parts >= min_part, up to max_parts parts."""
    results = []
    for num in range(1, max_parts + 1):
        for comp in itertools.product(range(min_part, total + 1), repeat=num):
            if sum(comp) == total:
                results.append(comp)
    return results

def _search_splits(deadline):
    n, m, p = DIMENSIONS
    best = None
    best_rank = n * m * p
    
    comp_i = _all_compositions(n)
    comp_j = _all_compositions(m)
    comp_k = _all_compositions(p)
    
    for ci in comp_i:
        for cj in comp_j:
            for ck in comp_k:
                if time.time() > deadline - 5:
                    break
                for use_str in [True, False]:
                    if time.time() > deadline - 5:
                        break
                    try:
                        result = _construct_hybrid(ci, cj, ck, use_str)
                        rank = result["_rank"]
                        if _verify(result["U"], result["V"], result["W"]):
                            if rank < best_rank:
                                best_rank = rank
                                best = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    except Exception:
                        pass
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 15:
                break
        if time.time() > deadline - 20:
            break
            
    return best

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 100
    best_result = None
    best_rank = 106
    
    # 1. Try schoolbook
    schoolbook = _construct_schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        best_result = schoolbook
        best_rank = 105
    
    # 2. Try systematic hybrid search
    if time.time() < deadline - 10:
        hybrid = _search_splits(deadline)
        if hybrid is not None and _verify(hybrid["U"], hybrid["V"], hybrid["W"]):
            r = len(hybrid["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = hybrid
    
    if best_result is None:
        return schoolbook
    return best_result
