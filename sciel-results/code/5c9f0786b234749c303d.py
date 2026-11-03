import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    The tensor identity is sum_r W[k*2+i, r] U[i*2+j, r] V[j*2+k, r] = delta_{k,i}
    for the standard matrix multiplication where C[i, k] = sum_j A[i, j] B[j, k].
    Note: The output index in W is defined as k*n+i. In this 2x2 block, n=2.
    So W[k*2+i] corresponds to the element C[i, k].
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] += 1
    W[3][0] += 1

    # Term 1: M2 = (A10 + A11)B00
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] += 1
    W[3][1] += -1

    # Term 2: M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] += 1
    W[3][2] += 1

    # Term 3: M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] += -1
    W[2][3] += 1

    # Term 4: M5 = (A00 + A01)B11
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] += 1
    W[2][4] = 1

    # Term 5: M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] += 1

    # Term 6: M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] += -1

    return U, V, W

def _get_splits(dim):
    """Get all possible compositions of dim into parts of size 1 or 2."""
    results = []
    def helper(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if remaining >= 1:
            current.append(1)
            helper(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            helper(remaining - 2, current)
            current.pop()
    helper(dim, [])
    return results

def _compute_rank_for_splits(i_split, j_split, k_split):
    """Compute the rank for given splits."""
    total = 0
    for i_sz in i_split:
        for j_sz in j_split:
            for k_sz in k_split:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    total += 7
                else:
                    total += i_sz * j_sz * k_sz
    return total

def _find_best_splits(n, m, p):
    """Find the best splits that minimize rank."""
    i_splits = _get_splits(n)
    j_splits = _get_splits(m)
    k_splits = _get_splits(p)
    
    best_rank = n * m * p
    best_splits = None
    
    for i_sp in i_splits:
        for j_sp in j_splits:
            for k_sp in k_splits:
                rank = _compute_rank_for_splits(i_sp, j_sp, k_sp)
                if rank < best_rank:
                    best_rank = rank
                    best_splits = (i_sp, j_sp, k_sp)
    
    return best_splits, best_rank

def _build_decomposition(n, m, p, i_split, j_split, k_split):
    """Build the decomposition for given splits."""
    su, sv, sw = _get_strassen_222()
    
    # Create index lists for each block
    i_blocks = []
    idx = 0
    for sz in i_split:
        i_blocks.append(list(range(idx, idx + sz)))
        idx += sz
    
    j_blocks = []
    idx = 0
    for sz in j_split:
        j_blocks.append(list(range(idx, idx + sz)))
        idx += sz
    
    k_blocks = []
    idx = 0
    for sz in k_split:
        k_blocks.append(list(range(idx, idx + sz)))
        idx += sz
    
    # Compute total rank
    total_rank = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b) == 2 and len(j_b) == 2 and len(k_b) == 2:
                    total_rank += 7
                else:
                    total_rank += len(i_b) * len(j_b) * len(k_b)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b) == 2 and len(j_b) == 2 and len(k_b) == 2:
                    # Strassen
                    for c in range(7):
                        for li in range(2):
                            gi = i_b[li]
                            for lj in range(2):
                                gj = j_b[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_b[lj]
                            for lk in range(2):
                                gk = k_b[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_b[lk]
                            for li in range(2):
                                gi = i_b[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_b:
                        for gj in j_b:
                            for gk in k_b:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _schoolbook_factors(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Find best splits
    best_splits, best_rank = _find_best_splits(n, m, p)
    
    if time.time() > deadline:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    i_split, j_split, k_split = best_splits
    
    # Build and verify
    U, V, W = _build_decomposition(n, m, p, i_split, j_split, k_split)
    rank = len(W[0])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
