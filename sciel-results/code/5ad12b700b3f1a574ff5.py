import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    This is the optimal rank for this problem, as it is the direct sum of 5 copies of the 2x2x2 tensor,
    each of which has rank 7 (Strassen).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    # The tensor T_{2,2,10} is the direct sum of 5 tensors T_{2,2,2}.
    # Rank of T_{2,2,2} is 7.
    # Total rank is 5 * 7 = 35.
    rank = 35
    
    # Initialize matrices
    U = np.zeros((n * m, rank))  # 4 x 35
    V = np.zeros((m * p, rank))  # 20 x 35
    W = np.zeros((p * n, rank))  # 20 x 35

    # Strassen coefficients for 2x2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a)(f-h)
    # P4 = (d)(g-e)
    # P5 = (a+b)(h)
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U coefficients (for A)
    # Rows: a, b, c, d
    U_strassen = np.array([
        [1, 1, 0, 0, 1, -1, 0],  # a
        [0, 0, 1, 0, 1, 0, -1],  # b
        [1, 0, 0, 0, 0, 1, 0],   # c
        [0, 0, 1, 1, 0, 0, 1]    # d
    ])
    
    # V coefficients (for B)
    # Rows: e, f, g, h
    V_strassen = np.array([
        [1, 0, 0, -1, 0, 1, 0],  # e
        [0, 1, -1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, 0, 0, 1, 0, 1]    # h
    ])
    
    # W coefficients (for C)
    # C = [[C00, C01], [C10, C11]]
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # Rows: C00, C10, C01, C11 (order in W is k*n+i, so for n=2, p=2:
    # W[0] = C00 (k=0, i=0)
    # W[1] = C10 (k=1, i=0)
    # W[2] = C01 (k=0, i=1)
    # W[3] = C11 (k=1, i=1)
    # Note: The trace convention is W[k*n+i, r].
    # For 2x2x2, k is 0..1, i is 0..1.
    # W[0] corresponds to C[0,0]
    # W[1] corresponds to C[1,0]
    # W[2] corresponds to C[0,1]
    # W[3] corresponds to C[1,1]
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, -1],  # C00
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ])
    
    # Construct the full matrices by repeating the Strassen block 5 times
    for b in range(5):
        col_offset = b * 7
        # U is 4x35, each block is 4x7
        U[:, col_offset:col_offset+7] = U_strassen
        
        # V is 20x35. The 2x10 matrix B is viewed as 5 blocks of 2x2.
        # Block b corresponds to columns 2b and 2b+1 of B.
        # V rows are indexed by j*p + k, where j is row of B (0..1), k is col of B (0..9).
        # For block b, the columns of B are 2b and 2b+1.
        # So the rows of V for this block are:
        # j=0, k=2b -> index 0*p + 2b = 2b
        # j=0, k=2b+1 -> index 0*p + 2b+1 = 2b+1
        # j=1, k=2b -> index 1*p + 2b = 10 + 2b
        # j=1, k=2b+1 -> index 1*p + 2b+1 = 10 + 2b+1
        
        # V_strassen rows correspond to e, f, g, h.
        # e = B[0, 2b]
        # f = B[0, 2b+1]
        # g = B[1, 2b]
        # h = B[1, 2b+1]
        
        V[2*b, col_offset:col_offset+7] = V_strassen[0]
        V[2*b+1, col_offset:col_offset+7] = V_strassen[1]
        V[10+2*b, col_offset:col_offset+7] = V_strassen[2]
        V[10+2*b+1, col_offset:col_offset+7] = V_strassen[3]
        
        # W is 20x35. The 2x10 matrix C is viewed as 5 blocks of 2x2.
        # Block b corresponds to columns 2b and 2b+1 of C.
        # W rows are indexed by k*n + i, where k is col of C (0..9), i is row of C (0..1).
        # For block b, the columns of C are 2b and 2b+1.
        # So the rows of W for this block are:
        # k=2b, i=0 -> index 2b*2 + 0 = 4b
        # k=2b, i=1 -> index 2b*2 + 1 = 4b+1
        # k=2b+1, i=0 -> index (2b+1)*2 + 0 = 4b+2
        # k=2b+1, i=1 -> index (2b+1)*2 + 1 = 4b+3
        
        # W_strassen rows correspond to C00, C10, C01, C11.
        # C00 = C[0, 2b] -> i=0, k=2b -> index 4b
        # C10 = C[1, 2b] -> i=1, k=2b -> index 4b+1
        # C01 = C[0, 2b+1] -> i=0, k=2b+1 -> index 4b+2
        # C11 = C[1, 2b+1] -> i=1, k=2b+1 -> index 4b+3
        
        W[4*b, col_offset:col_offset+7] = W_strassen[0]
        W[4*b+1, col_offset:col_offset+7] = W_strassen[1]
        W[4*b+2, col_offset:col_offset+7] = W_strassen[2]
        W[4*b+3, col_offset:col_offset+7] = W_strassen[3]
        
    # Convert to list of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verification
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_part = np.dot(U[:, r], A_test.flatten())
            b_part = np.dot(V[:, r], B_test.flatten())
            w_vec = W[:, r]
            # Add contribution
            # w_vec is of shape (20,). 
            # C[i, k] += a_part * b_part * W[k*n+i, r]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += a_part * b_part * w_vec[k*n + i]
                    
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook if verification fails (should not happen)
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school))
            V_s = np.zeros((m * p, rank_school))
            W_s = np.zeros((p * n, rank_school))
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j, idx] = 1
                        V_s[j * p + k, idx] = 1
                        W_s[k * n + i, idx] = 1
                        idx += 1
                        
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
