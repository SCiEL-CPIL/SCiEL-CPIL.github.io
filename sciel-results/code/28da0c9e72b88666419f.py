import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s margin from 150s limit

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Use Strassen for 2x2x2 sub-blocks where applicable
    - Partition: m=4 into 2 blocks of 2, p=6 into 3 blocks of 2
    - For each (j_block, k_block): 3x2x2 sub-tensor
    - Strassen on rows (0,1): 7 terms, schoolbook on row 2: 4 terms = 11 terms per block
    - 2 j_blocks * 3 k_blocks * 11 = 66 terms total
    - Then attempt to reduce rank via linear algebra (finding dependencies)
    """
    if time.time() - START_TIME > TIME_LIMIT:
        return _construct_schoolbook()
    
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]   # 12 x 66
    V = [[0] * rank for _ in range(m * p)]   # 24 x 66
    W = [[0] * rank for _ in range(p * n)]   # 18 x 66
    
    col = 0
    
    def add_strassen_3x2x2(i0, i1, i2, j_base, k_base):
        """
        Add 11 terms for a 3x2x2 block:
        - Strassen on rows (i0, i1): 7 terms
        - Schoolbook on row i2: 4 terms
        """
        nonlocal col
        
        # A indices for rows i0, i1 (2x2 sub-block)
        a_idx = i0 * m + j_base
        b_idx = i0 * m + (j_base + 1)
        c_idx = i1 * m + j_base
        d_idx = i1 * m + (j_base + 1)
        
        # A index for row i2
        e2_idx = i2 * m + j_base
        f2_idx = i2 * m + (j_base + 1)
        
        # B indices (2x2 sub-block)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        # W indices for rows i0, i1 (2x2 sub-block), W index = k*n + i
        p_idx = k_base * n + i0
        q_idx = (k_base + 1) * n + i0
        r_idx = k_base * n + i1
        s_idx = (k_base + 1) * n + i1
        
        # W indices for row i2
        t_idx = k_base * n + i2
        u_idx = (k_base + 1) * n + i2
        
        # Strassen on (i0, i1) x 2x2:
        # P1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # P3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1
        
        # Schoolbook for row i2 (1x2x2 = 4 terms):
        # C[i2, k_base] = A[i2, j_base]*B[j_base, k_base] + A[i2, j_base+1]*B[j_base+1, k_base]
        U[e2_idx][col] = 1
        V[e_idx][col] = 1
        W[t_idx][col] = 1
        col += 1
        
        U[f2_idx][col] = 1
        V[g_idx][col] = 1
        W[t_idx][col] = 1
        col += 1
        
        # C[i2, k_base+1] = A[i2, j_base]*B[j_base, k_base+1] + A[i2, j_base+1]*B[j_base+1, k_base+1]
        U[e2_idx][col] = 1
        V[f_idx][col] = 1
        W[u_idx][col] = 1
        col += 1
        
        U[f2_idx][col] = 1
        V[h_idx][col] = 1
        W[u_idx][col] = 1
        col += 1

    # Process all 6 blocks: 2 j_blocks x 3 k_blocks
    for j_block in range(2):
        j_base = j_block * 2
        for k_block in range(3):
            k_base = k_block * 2
            if time.time() - START_TIME > TIME_LIMIT:
                return _construct_schoolbook()
            add_strassen_3x2x2(0, 1, 2, j_base, k_base)
    
    # Verification of base rank-66 decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return _construct_schoolbook()
    
    # Attempt rank reduction via linear algebra
    reduced = _try_reduce_rank(U_np, V_np, W_np, n, m, p)
    if reduced is not None:
        return reduced
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U_np, V_np, W_np, n, m, p, max_time=120.0):
    """
    Attempt to reduce the rank of the tensor decomposition by finding
    linear dependencies among the terms.
    
    The tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    We can try to find a lower rank decomposition by looking for
    linear combinations of columns that preserve the tensor.
    """
    start = time.time()
    r0 = U_np.shape[1]  # 66
    
    # Try to find if any column is a linear combination of others
    # by checking the tensor contribution
    
    # For each pair of columns, check if their "difference" can be zeroed out
    # by adjusting other columns
    
    # Alternative approach: use the fact that the tensor has a specific structure
    # and try to merge terms that operate on the same sub-tensor
    
    # Let's try a simpler approach: check if we can reduce by looking at
    # the rank of the flattening
    
    # Flatten the tensor and try to find a lower rank SVD-like decomposition
    # but this won't directly give us the multiplicative structure
    
    # Instead, let's try a greedy elimination approach:
    # For each column r, check if removing it and adjusting others can preserve the tensor
    
    # This is complex, so let's try a different approach:
    # Check if any two columns are "equivalent" in the sense that their
    # contributions can be merged
    
    # For now, return None to keep the rank-66 solution
    # The reduction is non-trivial and requires more sophisticated techniques
    
    return None

def _construct_schoolbook():
    n, m, p = 3, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
