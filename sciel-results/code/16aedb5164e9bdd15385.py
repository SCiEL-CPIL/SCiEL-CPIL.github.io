import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U, V, W are (4, 7)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return the rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1.
    U (4,4), V (2,4), W (2,4)
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    
    # C1 = A11*B1 + A12*B2
    # A11 is U[0], B1 is V[0], W[0] (C1 part)
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # A12 is U[1], B2 is V[1], W[0] (C1 part)
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # C2 = A21*B1 + A22*B2
    # A21 is U[2], B1 is V[0], W[1] (C2 part)
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # A22 is U[3], B2 is V[1], W[1] (C2 part)
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _tensor_product_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Tensor product of two matrix multiplication decompositions.
    Returns U, V, W for <n1*n2, m1*m2, p1*p2>
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)
    
    # Pre-calculate indices for efficiency
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    # U index: i_new * m_new + j_new
                    i_new = i1 * n2 + i2
                    j_new = j1 * m2 + j2
                    u_idx = i_new * m + j_new
                    u1_val = U1[i1 * m1 + j1]
                    u2_val = U2[i2 * m2 + j2]
                    # U_new[:, r1*r2 + r2] = u1_val * u2_val
                    # We can vectorize the column assignment
                    prod = u1_val * u2_val
                    # r1 is fixed, r2 varies
                    # Columns are r1 * r2 + r2.
                    # So for each r1, we take U1[i1*m1+j1, r1] * U2[i2*m2+j2, :]
                    # This is a row of U1 times the matrix U2? No, element-wise.
                    # Actually, for a fixed (i1,j1,i2,j2), the value in column r1*r2+r2 is U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2].
                    # We can just loop r1 and r2, or use broadcasting.
                    for r1_idx in range(r1):
                        # U1_val is scalar
                        u1_scalar = U1[i1 * m1 + j1, r1_idx]
                        if u1_scalar == 0:
                            continue
                        # u2_row is vector of length r2
                        u2_row = U2[i2 * m2 + j2]
                        # prod_row = u1_scalar * u2_row
                        # Place into U[u_idx, r1_idx*r2 : (r1_idx+1)*r2]
                        U[u_idx, r1_idx * r2 : (r1_idx + 1) * r2] = u1_scalar * u2_row

    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    v_new = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                    v1_val = V1[j1 * p1 + k1]
                    v2_val = V2[j2 * p2 + k2]
                    for r1_idx in range(r1):
                        v1_scalar = V1[j1 * p1 + k1, r1_idx]
                        if v1_scalar == 0:
                            continue
                        v2_row = V2[j2 * p2 + k2]
                        V[v_new, r1_idx * r2 : (r1_idx + 1) * r2] = v1_scalar * v2_row

    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    w_new = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                    w1_val = W1[k1 * n1 + i1]
                    w2_val = W2[k2 * n2 + i2]
                    for r1_idx in range(r1):
                        w1_scalar = W1[k1 * n1 + i1, r1_idx]
                        if w1_scalar == 0:
                            continue
                        w2_row = W2[k2 * n2 + i2]
                        W[w_new, r1_idx * r2 : (r1_idx + 1) * r2] = w1_scalar * w2_row

    return U, V, W

def _construct_rank_77():
    """
    Construct the rank-77 decomposition.
    Part 1: 4x4x4 using Strassen (Rank 49)
    Part 2: 4x4x2 using Strassen x Schoolbook (Rank 28)
    """
    n, m, p = 4, 4, 6
    
    # Part 1: 4x4x4
    U_str, V_str, W_str = _strassen_2x2x2()
    U_444, V_444, W_444 = _tensor_product_decomp(
        U_str, V_str, W_str, 
        U_str, V_str, W_str, 
        2, 2, 2, 
        2, 2, 2
    )
    rank_444 = U_444.shape[1] # 49
    
    # Part 2: 4x4x2
    U_sch, V_sch, W_sch = _schoolbook_2x2x1()
    U_442, V_442, W_442 = _tensor_product_decomp(
        U_str, V_str, W_str, 
        U_sch, V_sch, W_sch, 
        2, 2, 2, 
        2, 2, 1
    )
    rank_442 = U_442.shape[1] # 28
    
    total_rank = rank_444 + rank_442 # 77
    
    U_total = np.zeros((n * m, total_rank), dtype=int)
    V_total = np.zeros((m * p, total_rank), dtype=int)
    W_total = np.zeros((p * n, total_rank), dtype=int)
    
    # Fill Part 1
    U_total[:, :rank_444] = U_444
    
    # V_444 indices: j (0..3), k (0..3). Shape 16x49.
    # V_total indices: j (0..3), k (0..5). Shape 24x77.
    # Map V_444[j*4+k] to V_total[j*6+k]
    for j in range(4):
        for k in range(4):
            V_total[j * p + k, :rank_444] = V_444[j * 4 + k]
            
    # W_444 indices: k (0..3), i (0..3). Shape 16x49.
    # W_total indices: k (0..5), i (0..3). Shape 24x77.
    # Map W_444[k*4+i] to W_total[k*4+i]
    for k in range(4):
        for i in range(4):
            W_total[k * n + i, :rank_444] = W_444[k * 4 + i]
            
    # Fill Part 2
    offset = rank_444
    U_total[:, offset:offset+rank_442] = U_442
    
    # V_442 indices: j (0..3), k (0..1). Shape 8x28. (p=2)
    # V_total indices: j (0..3), k (4..5).
    # Map V_442[j*2+k] to V_total[j*6+(4+k)]
    for j in range(4):
        for k in range(2):
            V_total[j * p + (4 + k), offset:offset+rank_442] = V_442[j * 2 + k]
            
    # W_442 indices: k (0..1), i (0..3). Shape 8x28. (n=4)
    # W_total indices: k (4..5), i (0..3).
    # Map W_442[k*4+i] to W_total[(4+k)*4+i]
    for k in range(2):
        for i in range(4):
            W_total[(4 + k) * n + i, offset:offset+rank_442] = W_442[k * 4 + i]
            
    return U_total, V_total, W_total

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _construct_rank_77()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
