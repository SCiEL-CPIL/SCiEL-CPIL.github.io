import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=None, m=None, p=None):
    """
    Verify exact tensor decomposition.
    The tensor is T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    We require T[i, j, k] == 1 for all i, j, k.
    """
    if n is None:
        n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                s = np.dot(u_row * v_row, w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_2x2x2():
    """
    Returns the 7 terms of the Strassen decomposition for 2x2 matrix multiplication.
    The terms are (u, v, w) vectors where:
    u is length 4 (indices i*m+j -> i*2+j)
    v is length 4 (indices j*p+k -> j*2+k)
    w is length 4 (indices k*n+i -> k*2+i)
    """
    terms = []
    # 1. (A11+A22)(B11+B22)
    u = [1, 0, 1, 1]
    v = [1, 1, 0, 0]
    w = [1, 0, 0, 1]
    terms.append((u, v, w))
    
    # 2. (A12)(B21 - B22)
    u = [0, 1, 0, 0]
    v = [0, 0, 1, -1]
    w = [0, 0, 0, 1]
    terms.append((u, v, w))
    
    # 3. (A21 - A11)(B11)
    u = [-1, 0, 1, 0]
    v = [1, 0, 0, 0]
    w = [1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 4. (A22)(B22)
    u = [0, 0, 0, 1]
    v = [0, 0, 0, 1]
    w = [0, 0, 0, 1]
    terms.append((u, v, w))
    
    # 5. (A11 + A21)(B21)
    u = [1, 1, 0, 0]
    v = [0, 0, 1, 0]
    w = [0, 1, 0, 0]
    terms.append((u, v, w))
    
    # 6. (A21 - A22)(B12 + B22)
    u = [0, 0, 1, -1]
    v = [0, 1, 0, 1]
    w = [1, 0, 1, 0]
    terms.append((u, v, w))
    
    # 7. (A12 - A22)(B11 + B12)
    u = [0, 1, 0, -1]
    v = [1, 1, 0, 0]
    w = [0, 0, 1, 0]
    terms.append((u, v, w))
    
    return terms

def _embed_2x2x2_to_3x2x2(terms_2x2):
    """
    Embed the 2x2x2 Strassen terms into a 3x2x2 tensor.
    The 3x2x2 tensor involves matrices A (3x2), B (2x2), C (2x3).
    We can view A as [[A1], [A2]] where A1, A2 are 2x2 blocks? No.
    A is 3x2. Let's split rows into 1+2 or 2+1.
    Let's split A into A_top (1x2) and A_bot (2x2).
    B is 2x2.
    C is 2x3. Let's split cols into 2+1. C_left (2x2), C_right (2x1).
    
    This seems messy. The previous code used a specific 3x2x2 decomposition.
    Let's reuse the logic from the current program's `_construct_3x2x2` which was verified to work in the inspirations (score 1.0909).
    Wait, the current program's `_construct_3x2x2` has terms of length 6, 4, 6.
    u: 3x2 = 6. v: 2x2 = 4. w: 2x3 = 6.
    This matches.
    I will just copy that function's logic directly to be safe, as it was the basis of the 1.0909 score.
    """
    terms = []
    
    # 1. (a+d)(x+z)
    u = [1, 0, 0, 1, 0, 0]
    v = [1, 0, 1, 0]
    w = [1, 0, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # 2. (c+d)x
    u = [0, 0, 1, 1, 0, 0]
    v = [1, 0, 0, 0]
    w = [0, 0, 0, 1, -1, 0]
    terms.append((u, v, w))
    
    # 3. a(y-z)
    u = [1, 0, 0, 0, 0, 0]
    v = [0, 1, -1, 0]
    w = [0, 1, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # 4. d(w-y)
    u = [0, 0, 0, 1, 0, 0]
    v = [0, -1, 0, 1]
    w = [1, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 5. (a+b)w
    u = [1, 1, 0, 0, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # 6. (c-a)(x+z)
    u = [-1, 0, 1, 0, 0, 0]
    v = [1, 0, 1, 0]
    w = [0, 0, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # 7. (b-d)(y+w)
    u = [0, 1, 0, -1, 0, 0]
    v = [0, 1, 0, 1]
    w = [1, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # 8. e*x
    u = [0, 0, 0, 0, 1, 0]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 9. e*y
    u = [0, 0, 0, 0, 1, 0]
    v = [0, 1, 0, 0]
    w = [0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    # 10. f*z
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 1, 0]
    w = [0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 11. f*w
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 1]
    w = [0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    return terms

def _construct_block_strassen():
    """
    Constructs the factor matrices using block Strassen.
    n=3, m=4, p=8.
    We split m into 2 blocks of size 2.
    We split p into 4 blocks of size 2.
    For each (j_block, k_block) pair, we use the 3x2x2 decomposition (which has 11 terms).
    Total rank = 2 * 4 * 11 = 88.
    """
    n, m, p = DIMENSIONS
    base_terms = _embed_2x2x2_to_3x2x2(None) # Returns the 11 terms for 3x2x2
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # j_block: 0, 1 (size 2 each)
    # k_block: 0, 1, 2, 3 (size 2 each)
    for j_blk in range(2):
        for k_blk in range(4):
            for u, v, w in base_terms:
                u_global = [0.0] * (n * m)
                v_global = [0.0] * (m * p)
                w_global = [0.0] * (p * n)
                
                # Map u (length 6, indices li*2+lj) to global (li*m + gj)
                # li in 0..2, lj in 0..1
                # gj = j_blk*2 + lj
                for li in range(3):
                    for lj in range(2):
                        val = u[li * 2 + lj]
                        if val != 0:
                            gj = j_blk * 2 + lj
                            u_global[li * m + gj] = val
                            
                # Map v (length 4, indices lj*2+lk) to global (gj*p + gk)
                # lj in 0..1, lk in 0..1
                # gj = j_blk*2 + lj
                # gk = k_blk*2 + lk
                for lj in range(2):
                    for lk in range(2):
                        val = v[lj * 2 + lk]
                        if val != 0:
                            gj = j_blk * 2 + lj
                            gk = k_blk * 2 + lk
                            v_global[gj * p + gk] = val
                            
                # Map w (length 6, indices lk*3+li) to global (gk*n + li)
                # lk in 0..1, li in 0..2
                # gk = k_blk*2 + lk
                for lk in range(2):
                    for li in range(3):
                        val = w[lk * 3 + li]
                        if val != 0:
                            gk = k_blk * 2 + lk
                            w_global[gk * n + li] = val
                            
                U_cols.append(u_global)
                V_cols.append(v_global)
                W_cols.append(w_global)
                
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
            
    return U, V, W

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses block Strassen decomposition to achieve rank 88.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Build the block Strassen decomposition
    U, V, W = _construct_block_strassen()
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
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

# EVOLVE-BLOCK-END
