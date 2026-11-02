import time
import numpy as np

# Dimensions for the matrix multiplication problem: (2, 2, 16)
N_DIM = 2
M_DIM = 2
P_DIM = 16

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = N_DIM, M_DIM, P_DIM
    
    # Number of 2x2 blocks in the 2x16 matrix
    num_blocks = p // 2
    
    # Rank of 2x2x2 multiplication is 7 (Strassen)
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    # Initialize factor matrices
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrices
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    u_terms = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: 1},   # M2
        {0: 1},         # M3
        {3: 1},         # M4
        {0: 1, 1: 1},   # M5
        {2: 1, 0: -1},  # M6
        {1: 1, 3: -1}   # M7
    ]
    
    v_terms = [
        {0: 1, 3: 1},   # M1
        {0: 1},         # M2
        {1: 1, 3: -1},  # M3
        {2: 1, 0: -1},  # M4
        {3: 1},         # M5
        {0: 1, 1: 1},   # M6
        {2: 1, 3: 1}    # M7
    ]
    
    w_terms = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: -1},  # M2
        {1: 1, 3: 1},   # M3
        {0: 1, 2: 1},   # M4
        {0: -1, 1: 1},  # M5
        {3: 1},         # M6
        {0: 1}          # M7
    ]
    
    # Fill the matrices for each block
    for b_idx in range(num_blocks):
        # Column indices in B for this block
        b_col_base = 2 * b_idx
        # Row indices in A are 0, 1
        # B indices for this block:
        # B[0, b_col_base] -> local 0
        # B[0, b_col_base+1] -> local 1
        # B[1, b_col_base] -> local 2
        # B[1, b_col_base+1] -> local 3
        
        # C indices for this block:
        # C[0, b_col_base] -> local 0
        # C[0, b_col_base+1] -> local 1
        # C[1, b_col_base] -> local 2
        # C[1, b_col_base+1] -> local 3
        
        for s in range(rank_per_block):
            r_idx = b_idx * rank_per_block + s
            
            # Update U
            for a_local, coeff in u_terms[s].items():
                # A index: i*m + j. Here i,j are local A indices?
                # Wait, U shape is (n*m, R). n=2, m=2.
                # Index 0: A[0,0], 1: A[0,1], 2: A[1,0], 3: A[1,1].
                # This matches the local indices 0,1,2,3.
                U[a_local, r_idx] = coeff
            
            # Update V
            for b_local, coeff in v_terms[s].items():
                # V shape is (m*p, R). m=2, p=16.
                # Index: j*p + k.
                # j=0, k=b_col_base -> b_local 0
                # j=0, k=b_col_base+1 -> b_local 1
                # j=1, k=b_col_base -> b_local 2
                # j=1, k=b_col_base+1 -> b_local 3
                if b_local == 0:
                    v_idx = 0 * p + b_col_base
                elif b_local == 1:
                    v_idx = 0 * p + (b_col_base + 1)
                elif b_local == 2:
                    v_idx = 1 * p + b_col_base
                elif b_local == 3:
                    v_idx = 1 * p + (b_col_base + 1)
                else:
                    continue
                V[v_idx, r_idx] = coeff
            
            # Update W
            for c_local, coeff in w_terms[s].items():
                # W shape is (p*n, R). p=16, n=2.
                # Index: k*n + i.
                # k is column index in C, i is row index in C.
                # c_local 0: C[0, b_col_base] -> i=0, k=b_col_base
                # c_local 1: C[0, b_col_base+1] -> i=0, k=b_col_base+1
                # c_local 2: C[1, b_col_base] -> i=1, k=b_col_base
                # c_local 3: C[1, b_col_base+1] -> i=1, k=b_col_base+1
                
                if c_local == 0:
                    i_val = 0
                    k_val = b_col_base
                elif c_local == 1:
                    i_val = 0
                    k_val = b_col_base + 1
                elif c_local == 2:
                    i_val = 1
                    k_val = b_col_base
                elif c_local == 3:
                    i_val = 1
                    k_val = b_col_base + 1
                else:
                    continue
                
                w_idx = k_val * n + i_val
                W[w_idx, r_idx] = coeff
    
    # Quick verification
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        # Compute decomposition
        # U is (n*m, R), A_flat is (n*m,). UA = U.T @ A_flat
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        products = UA * VB
        C_decomposed = W @ products
        
        # C_decomposed is (p*n,). C_expected is (n, p).
        # W index k*n + i corresponds to C[i, k].
        # So C_decomposed[k*n + i] = C_expected[i, k].
        # C_expected.T.flatten() will be in order k*n + i (row-major of transposed matrix).
        C_expected_flat = C_expected.T.flatten()
        
        if not np.array_equal(C_decomposed, C_expected_flat):
            # Fallback to schoolbook if verification fails
            # This should not happen, but safe to have
            rank_fallback = n * m * p
            U = np.zeros((n * m, rank_fallback), dtype=int)
            V = np.zeros((m * p, rank_fallback), dtype=int)
            W = np.zeros((p * n, rank_fallback), dtype=int)
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, col] = 1
                        V[v_idx, col] = 1
                        W[w_idx, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
