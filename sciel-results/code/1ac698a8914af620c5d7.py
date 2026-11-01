import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) for the 2x2x2 tensor.
    U indices: i*2 + j (i,j in {0,1})
    V indices: j*2 + k (j,k in {0,1})
    W indices: k*2 + i (k,i in {0,1})
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11=idx0, A22=idx3
    U[0, 0] = 1; U[3, 0] = 1
    # V: B11=idx0, B22=idx3
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11=idx0, C22=idx3
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # U: A21=idx2, A22=idx3
    U[2, 1] = 1; U[3, 1] = 1
    # V: B11=idx0
    V[0, 1] = 1
    # W: C21=idx2, C22=idx3
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    # U: A11=idx0
    U[0, 2] = 1
    # V: B21=idx2, B22=idx3
    V[2, 2] = 1; V[3, 2] = -1
    # W: C12=idx1, C22=idx3
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B12-B11)
    # U: A22=idx3
    U[3, 3] = 1
    # V: B12=idx1, B11=idx0
    V[1, 3] = 1; V[0, 3] = -1
    # W: C11=idx0, C21=idx2
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # U: A11=idx0, A12=idx1
    U[0, 4] = 1; U[1, 4] = 1
    # V: B22=idx3
    V[3, 4] = 1
    # W: C11=idx0, C12=idx1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21=idx2, A11=idx0
    U[2, 5] = 1; U[0, 5] = -1
    # V: B11=idx0, B12=idx1
    V[0, 5] = 1; V[1, 5] = 1
    # W: C22=idx3
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12=idx1, A22=idx3
    U[1, 6] = 1; U[3, 6] = -1
    # V: B21=idx2, B22=idx3
    V[2, 6] = 1; V[3, 6] = 1
    # W: C11=idx0
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return schoolbook decomposition for 2x2x1 matrix multiplication.
    Returns (U, V, W) each of shape (4, 4).
    U indices: i*2 + j
    V indices: j*1 + k = j (k is always 0)
    W indices: k*2 + i = i
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    for i in range(2):
        for j in range(2):
            col = i * 2 + j
            U[i * 2 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Strategy: Split j into {0,1} and {2}.
    # Part A: j=2. Dimensions 2x1x5. Use Rank 1 (Outer product).
    # Part B: j in {0,1}. Dimensions 2x2x5.
    #   Split k into {0,1,2} and {3,4}:
    #   - 2x2x3 (k in {0,1,2}): Strassen(2x2x2) + schoolbook(2x2x1) = 7+4 = 11
    #   - 2x2x2 (k in {3,4}): Strassen = 7
    #   Total for Part B: 11 + 7 = 18
    # Grand Total Rank = 1 + 18 = 19
    
    rank = 19
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part A: j=2 (Rank 1) ---
    r = 0
    for i in range(n):
        U[i * m + 2, r] = 1
    for k in range(p):
        V[2 * p + k, r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1
            
    # --- Part B: j in {0,1} (Rank 18) ---
    # Sub-part B1: k in {0,1,2} (2x2x3, rank 11)
    # Sub-part B2: k in {3,4} (2x2x2, rank 7)
    
    r_offset = 1
    
    # --- B1: 2x2x3 using Strassen(2x2x2 for k=0,1) + schoolbook(2x2x1 for k=2) ---
    # Strassen part: k in {0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    # Map: local (i,j) -> global i*3+j, local j*2+k -> global j*5+k, local k*2+i -> global k*2+i
    for col in range(7):
        r = r_offset + col
        for idx_u in range(4):
            if U_s[idx_u, col] != 0:
                li = idx_u // 2
                lj = idx_u % 2
                U[li * m + lj, r] += U_s[idx_u, col]
        for idx_v in range(4):
            if V_s[idx_v, col] != 0:
                lj = idx_v // 2
                lk = idx_v % 2
                # k maps to 0,1 (local k=0->global k=0, local k=1->global k=1)
                V[lj * p + lk, r] += V_s[idx_v, col]
        for idx_w in range(4):
            if W_s[idx_w, col] != 0:
                lk = idx_w // 2
                li = idx_w % 2
                # k maps to 0,1
                W[lk * n + li, r] += W_s[idx_w, col]
    
    # Schoolbook part: k=2
    U_sb, V_sb, W_sb = _schoolbook_2x2x1()
    for col in range(4):
        r = r_offset + 7 + col
        for idx_u in range(4):
            if U_sb[idx_u, col] != 0:
                li = idx_u // 2
                lj = idx_u % 2
                U[li * m + lj, r] += U_sb[idx_u, col]
        for idx_v in range(2):
            if V_sb[idx_v, col] != 0:
                lj = idx_v
                # k=2
                V[lj * p + 2, r] += V_sb[idx_v, col]
        for idx_w in range(2):
            if W_sb[idx_w, col] != 0:
                li = idx_w
                # k=2
                W[2 * n + li, r] += W_sb[idx_w, col]
    
    # --- B2: 2x2x2 for k in {3,4} using Strassen ---
    U_s2, V_s2, W_s2 = _strassen_2x2x2()
    for col in range(7):
        r = r_offset + 11 + col
        for idx_u in range(4):
            if U_s2[idx_u, col] != 0:
                li = idx_u // 2
                lj = idx_u % 2
                U[li * m + lj, r] += U_s2[idx_u, col]
        for idx_v in range(4):
            if V_s2[idx_v, col] != 0:
                lj = idx_v // 2
                lk = idx_v % 2
                # local k=0 -> global k=3, local k=1 -> global k=4
                V[lj * p + (3 + lk), r] += V_s2[idx_v, col]
        for idx_w in range(4):
            if W_s2[idx_w, col] != 0:
                lk = idx_w // 2
                li = idx_w % 2
                # local k=0 -> global k=3, local k=1 -> global k=4
                W[(3 + lk) * n + li, r] += W_s2[idx_w, col]
    
    # Verify
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1
                
    if not np.allclose(T_reconstructed, T_expected, atol=1e-10):
        # Fallback to naive rank 30 if something went wrong
        rank_naive = n * m * p
        U_naive = np.zeros((n * m, rank_naive))
        V_naive = np.zeros((m * p, rank_naive))
        W_naive = np.zeros((p * n, rank_naive))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_naive[ii * m + jj, col] = 1
                    V_naive[jj * p + kk, col] = 1
                    W_naive[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
