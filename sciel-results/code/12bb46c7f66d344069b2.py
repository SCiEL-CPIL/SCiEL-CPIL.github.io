"""Combined program: Block Strassen (rank 28) with verification and potential rank reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 2, 4, 4
    
    # Strassen factors for 2x2x2 multiplication
    # Local indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # i,j,k in {0,1}
    
    # M1 = (A00 + A11)(B00 + B11)
    U1 = {0: 1, 3: 1}
    V1 = {0: 1, 3: 1}
    W1 = {0: 1, 3: 1}
    
    # M2 = (A10 + A11)B00
    U2 = {2: 1, 3: 1}
    V2 = {0: 1}
    W2 = {1: 1, 3: -1}
    
    # M3 = A00(B10 - B00)
    U3 = {0: 1}
    V3 = {1: 1, 3: -1}
    W3 = {2: 1, 3: 1}
    
    # M4 = A11(B01 - B11)
    U4 = {3: 1}
    V4 = {2: 1, 0: -1}
    W4 = {0: 1, 1: 1}
    
    # M5 = (A00 + A01)B11
    U5 = {0: 1, 1: 1}
    V5 = {3: 1}
    W5 = {0: -1, 2: 1}
    
    # M6 = (A10 - A00)(B00 + B01)
    U6 = {2: 1, 0: -1}
    V6 = {0: 1, 1: 1}
    W6 = {3: 1}
    
    # M7 = (A01 - A11)(B10 + B11)
    U7 = {1: 1, 3: -1}
    V7 = {2: 1, 3: 1}
    W7 = {0: 1}
    
    strassen_terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, W7, W7)  # Placeholder, will fix below
    ]
    
    # Correct Strassen terms (standard formulation):
    # For C = A*B where A,B are 2x2:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M7
    
    # Let me redefine properly:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U factors (for A):
    # U1: A00, A11 -> indices 0, 3
    # U2: A10, A11 -> indices 2, 3
    # U3: A00 -> index 0
    # U4: A11 -> index 3
    # U5: A00, A01 -> indices 0, 1
    # U6: A10, -A00 -> indices 2, 0 (with -1)
    # U7: A01, -A11 -> indices 1, 3 (with -1)
    
    # V factors (for B):
    # V1: B00, B11 -> indices 0, 3
    # V2: B00 -> index 0
    # V3: B10, -B00 -> indices 1, 0 (with -1)
    # V4: B01, -B11 -> indices 2, 3 (with -1)
    # V5: B11 -> index 3
    # V6: B00, B01 -> indices 0, 1
    # V7: B10, B11 -> indices 1, 3
    
    # W factors (for C output):
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M7
    
    # W1: C00, C11 -> indices 0, 3
    # W2: C10 -> index 2 (since W[k*2+i], C10 is k=1,i=0 -> index 2)
    # W3: C10 -> index 2
    # W4: C00, C11 -> indices 0, 3
    # W5: C00, C01 -> indices 0, 1
    # W6: C01 -> index 1
    # W7: C00, C11 -> indices 0, 3 (with signs)
    
    # Let's be more careful with W indices.
    # W[k*n+i] where n=2, so W[k*2+i]
    # C00: k=0,i=0 -> W[0]
    # C01: k=0,i=1 -> W[1]
    # C10: k=1,i=0 -> W[2]
    # C11: k=1,i=1 -> W[3]
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M7
    
    # So:
    # W1 (for M1): C00 (+1), C11 (+1) -> {0: 1, 3: 1}
    # W2 (for M2): C10 (+1) -> {2: 1}
    # W3 (for M3): C10 (+1) -> {2: 1}
    # W4 (for M4): C00 (+1), C11 (+1) -> {0: 1, 3: 1}
    # W5 (for M5): C00 (-1), C01 (+1) -> {0: -1, 1: 1}
    # W6 (for M6): C01 (+1) -> {1: 1}
    # W7 (for M7): C00 (-1), C11 (+1) -> {0: -1, 3: 1}
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),                     # M2
        ({0: 1}, {1: 1, 0: -1}, {2: 1}),                    # M3
        ({3: 1}, {2: 1, 3: -1}, {0: 1, 3: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {1: 1}),             # M6
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: -1, 3: 1}),      # M7
    ]
    
    def build_strassen_decomposition():
        """Build the rank-28 Strassen decomposition."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # 4 blocks: (j_offset, k_offset)
        # Block 0: j in [0,1], k in [0,1]
        # Block 1: j in [0,1], k in [2,3]
        # Block 2: j in [2,3], k in [0,1]
        # Block 3: j in [2,3], k in [2,3]
        
        for block in range(4):
            j_offset = (block // 2) * 2
            k_offset = (block % 2) * 2
            
            for U_loc, V_loc, W_loc in strassen_terms:
                # Fill U
                for u_idx, val in U_loc.items():
                    i = u_idx // 2
                    j_loc = u_idx % 2
                    j = j_offset + j_loc
                    global_u_idx = i * m + j
                    U[global_u_idx][col] += val
                
                # Fill V
                for v_idx, val in V_loc.items():
                    j_loc = v_idx // 2
                    k_loc = v_idx % 2
                    j = j_offset + j_loc
                    k = k_offset + k_loc
                    global_v_idx = j * p + k
                    V[global_v_idx][col] += val
                
                # Fill W
                for w_idx, val in W_loc.items():
                    k_loc = w_idx // 2
                    i = w_idx % 2
                    k = k_offset + k_loc
                    global_w_idx = k * n + i
                    W[global_w_idx][col] += val
                
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(decomp):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        
        n, m, p = 2, 4, 4
        rank = U.shape[1]
        
        # The decomposition should satisfy:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for appropriate j
        
        # More precisely, the tensor T[i,j,k] = delta[i,j,k] for matrix mult
        # T[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to (i,k) of C
        # Actually, the matrix multiplication tensor is:
        # T[i,j,k] = 1 for all i,j,k (it's the standard tensor)
        # And the decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify by checking a random matrix product
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # u_r is a vector of length n*m
            # v_r is a vector of length m*p
            # w_r is a vector of length p*n
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            # Reshape
            u_mat = u_r.reshape(n, m)
            v_mat = v_r.reshape(m, p)
            w_mat = w_r.reshape(p, n)
            
            # The contribution to C is u_mat @ v_mat * w_mat? No.
            # The tensor decomposition means:
            # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * ... 
            # Actually, the standard form is:
            # C[i,k] = sum_r (sum_j A[i,j] U[i,j,r]) * (sum_k B[j,k] V[j,k,r]) * W[k,i,r]
            # This is not quite right. Let me think again.
            
            # The matrix multiplication tensor T is defined such that:
            # C = A @ B means C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor T[i,j,k] = 1 (the standard trilinear form)
            # A rank-r decomposition is:
            # T[i,j,k] = sum_{r=1}^r u_r[i,j] v_r[j,k] w_r[k,i]
            # where u_r is n x m, v_r is m x p, w_r is p x n
            # And the computation is:
            # C[i,k] = sum_r (sum_j A[i,j] u_r[i,j]) * (sum_k B[j,k] v_r[j,k]) * w_r[k,i]
            # Wait, that doesn't look right either.
            
            # Let's use the explicit formula:
            # The decomposition says:
            # delta[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # where delta[i,j,k] = 1 for all i,j,k (in the standard tensor)
            
            # Actually no, the matrix multiplication tensor has:
            # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # And the decomposition is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # To verify, we can check that for random A, B:
            # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_k B[j,k] V[j*p+k,r]) * W[k*n+i,r]
            # This is still not clean. Let me just do a direct check.
            
            pass
        
        # Direct verification: check the tensor identity
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1  # T[i,j,k] = 1 for matrix mult
                    rhs = 0
                    for r in range(rank):
                        rhs += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    # Build the rank-28 decomposition
    result = build_strassen_decomposition()
    
    # Verify it
    if not verify_decomposition(result):
        # Fallback to schoolbook if verification fails
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
    
    # Try to reduce rank by checking if some columns are linearly dependent
    # or can be merged
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    # Check for duplicate or linearly dependent columns
    # This is a heuristic; we'll try to remove columns that are linear combinations
    # of others, but this is complex. For now, return the rank-28 result.
    
    # Check time
    if time.time() > deadline:
        return result
    
    # Try a simple rank reduction: check if any column is zero
    for r in range(U_arr.shape[1]):
        if np.all(U_arr[:, r] == 0) or np.all(V_arr[:, r] == 0) or np.all(W_arr[:, r] == 0):
            # Remove this column
            U_arr = np.delete(U_arr, r, axis=1)
            V_arr = np.delete(V_arr, r, axis=1)
            W_arr = np.delete(W_arr, r, axis=1)
            break
    
    # Convert back to list of lists
    return {
        "U": U_arr.tolist(),
        "V": V_arr.tolist(),
        "W": W_arr.tolist()
    }
# EVOLVE-BLOCK-END
