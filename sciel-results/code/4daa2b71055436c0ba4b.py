import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7) respectively.
    
    The 7 products are:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U factors: linear combinations of A entries
    # A is indexed as A[i,j] where i,j in {0,1}, flattened as i*2+j
    # U[u_idx, r] where u_idx = i*2+j
    U = [[0]*7 for _ in range(4)]
    # V factors: linear combinations of B entries  
    # B is indexed as B[i,j] where i,j in {0,1}, flattened as i*2+j
    V = [[0]*7 for _ in range(4)]
    # W factors: linear combinations for C entries
    # C is indexed as C[i,j] where i,j in {0,1}, flattened as i*2+j
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1  # A[0,0]
    U[3][0] = 1  # A[1,1]
    V[0][0] = 1  # B[0,0]
    V[3][0] = 1  # B[1,1]
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # M2 = (A21 + A22)B11
    U[2][1] = 1  # A[1,0]
    U[3][1] = 1  # A[1,1]
    V[0][1] = 1  # B[0,0]
    W[2][1] = 1  # C[1,0]
    W[3][1] = -1 # C[1,1]
    
    # M3 = A11(B21 - B22)
    U[0][2] = 1  # A[0,0]
    V[2][2] = 1  # B[1,0]
    V[3][2] = -1 # B[1,1]
    W[1][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # M4 = A22(B12 - B11)
    U[3][3] = 1  # A[1,1]
    V[1][3] = 1  # B[0,1]
    V[0][3] = -1 # B[0,0]
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    
    # M5 = (A11 + A12)B22
    U[0][4] = 1  # A[0,0]
    U[1][4] = 1  # A[0,1]
    V[3][4] = 1  # B[1,1]
    W[0][4] = -1 # C[0,0]
    W[1][4] = 1  # C[0,1]
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2][5] = 1  # A[1,0]
    U[0][5] = -1 # A[0,0]
    V[0][5] = 1  # B[0,0]
    V[1][5] = 1  # B[0,1]
    W[3][5] = 1  # C[1,1]
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1  # A[0,1]
    U[3][6] = -1 # A[1,1]
    V[2][6] = 1  # B[1,0]
    V[3][6] = 1  # B[1,1]
    W[0][6] = 1  # C[0,0]
    
    return U, V, W

def _block_decomposition():
    """
    Decompose 5x5x5 matmul using block structure.
    Partition indices into blocks: {0,1}, {2,3}, {4}
    
    Block sizes: 2, 2, 1
    Number of blocks: 3
    
    For each pair of row-block (ib), col-block (jb), and inner-block (kb),
    we need to compute C[ib,kb] += A[ib,jb] * B[jb,kb]
    
    We'll use:
    - Strassen (rank 7) for 2x2x2 blocks
    - Schoolbook (rank 1) for blocks involving the 1x1 dimension
    """
    n, m, p = DIMENSIONS  # all 5
    
    # Block boundaries: [0, 2, 4, 5]
    # Block sizes: [2, 2, 1]
    block_sizes = [2, 2, 1]
    block_starts = [0, 2, 4]
    num_blocks = 3
    
    # Collect all rank-1 terms
    terms = []  # Each term is (u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs)
    
    for ib in range(num_blocks):  # row block of A, row block of C
        for kb in range(num_blocks):  # col block of B, col block of C
            for jb in range(num_blocks):  # inner block (col of A, row of B)
                ib_size = block_sizes[ib]
                jb_size = block_sizes[jb]
                kb_size = block_sizes[kb]
                
                if ib_size == 2 and jb_size == 2 and kb_size == 2:
                    # Use Strassen: rank 7
                    U_s, V_s, W_s = _strassen_2x2()
                    for r in range(7):
                        u_idx = []
                        u_coef = []
                        for local_i in range(2):
                            for local_j in range(2):
                                global_i = block_starts[ib] + local_i
                                global_j = block_starts[jb] + local_j
                                u_idx.append(global_i * m + global_j)
                                u_coef.append(U_s[local_i * 2 + local_j][r])
                        
                        v_idx = []
                        v_coef = []
                        for local_i in range(2):
                            for local_j in range(2):
                                global_i = block_starts[jb] + local_i
                                global_j = block_starts[kb] + local_j
                                v_idx.append(global_i * p + global_j)
                                v_coef.append(V_s[local_i * 2 + local_j][r])
                        
                        w_idx = []
                        w_coef = []
                        for local_i in range(2):
                            for local_j in range(2):
                                global_i = block_starts[ib] + local_i
                                global_j = block_starts[kb] + local_j
                                w_idx.append(global_j * n + global_i)
                                w_coef.append(W_s[local_i * 2 + local_j][r])
                        
                        terms.append((u_idx, u_coef, v_idx, v_coef, w_idx, w_coef))
                else:
                    # Schoolbook for mixed-size blocks
                    for local_i in range(ib_size):
                        for local_j in range(jb_size):
                            for local_k in range(kb_size):
                                global_i = block_starts[ib] + local_i
                                global_j = block_starts[jb] + local_j
                                global_k = block_starts[kb] + local_k
                                
                                u_idx = [global_i * m + global_j]
                                u_coef = [1]
                                v_idx = [global_j * p + global_k]
                                v_coef = [1]
                                w_idx = [global_k * n + global_i]
                                w_coef = [1]
                                
                                terms.append((u_idx, u_coef, v_idx, v_coef, w_idx, w_coef))
    
    # Build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, u_coef, v_idx, v_coef, w_idx, w_coef) in enumerate(terms):
        for idx, coef in zip(u_idx, u_coef):
            if coef != 0:
                U[idx][r] = coef
        for idx, coef in zip(v_idx, v_coef):
            if coef != 0:
                V[idx][r] = coef
        for idx, coef in zip(w_idx, w_coef):
            if coef != 0:
                W[idx][r] = coef
    
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p):
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    # Verify by checking that the tensor decomposition matches matmul
    # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
    # where u = i*m + j, v = j'*p + k, w = k'*n + i'
    # Expected: T[u, v, w] = 1 if j == j' and k == k' and i == i', else 0
    
    # For efficiency, check a sample of points
    np.random.seed(42)
    num_checks = 50000
    for _ in range(num_checks):
        u = np.random.randint(0, n * m)
        v = np.random.randint(0, m * p)
        w = np.random.randint(0, p * n)
        
        i_u = u // m
        j_u = u % m
        j_v = v // p
        k_v = v % p
        k_w = w // n
        i_w = w % n
        
        expected = 1.0 if (j_u == j_v and k_v == k_w and i_w == i_u) else 0.0
        actual = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
        
        if abs(actual - expected) > 1e-9:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition with Strassen
    decomp = _block_decomposition()
    
    # Verify the decomposition
    if _verify(decomp, n, m, p):
        return decomp
    
    # Fallback to schoolbook if verification fails
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][r] = 1
                    V[v_idx][r] = 1
                    W[w_idx][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    return _schoolbook()
# EVOLVE-BLOCK-END
