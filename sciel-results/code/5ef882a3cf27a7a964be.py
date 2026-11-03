import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill

    # Build schoolbook rank-72 decomposition as starting point
    rank_sb = n * m * p  # 72
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb

    # Strategy: Iterative rank-1 peeling (deflation)
    # Start with the full tensor T (flattened) and greedily extract rank-1 components
    # T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    
    # Represent the tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Greedy rank-1 peeling with SVD
    T_residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    max_rank = 70  # Don't exceed schoolbook

    for r in range(max_rank):
        if time.time() > deadline - 20:
            break
        
        # Flatten along mode 2 to get a matrix: (n*m) x (m*p * p*n)
        # Actually, flatten modes 1 and 2: (n*m) x (m*p * p*n)
        mat = T_residual.reshape(n * m, m * p * p * n)
        
        # SVD
        try:
            U_s, s, Vt = np.linalg.svd(mat, full_matrices=False)
        except:
            break
        
        if s[0] < 1e-10:
            # Tensor is essentially zero - done
            break
        
        # The top singular vector gives the best rank-1 approximation direction
        # u1 is shape (n*m,), v1 is shape (m*p*p*n,)
        u1 = U_s[:, 0]
        v1 = Vt[0, :].reshape(m * p, p * n)
        
        # For exact rank-1: T = u1 ⊗ v1 (outer product)
        # Check if the residual after subtracting this rank-1 is still valid
        # Actually, for exact decomposition we need T = sum of rank-1 terms
        # Greedy: subtract the best rank-1 approximation
        
        # The best rank-1 approx of T_residual is s[0] * u1 ⊗ (v1 / s[0]) = s[0]*u1 ⊗ v1/s[0]
        # But we want exact: let's just use u1 and v1 as the factor vectors
        # T_rank1 = np.outer(u1, v1.flatten()).reshape(n*m, m*p, p*n)
        
        # Subtract
        T_rank1 = np.outer(u1, v1.flatten()).reshape(n * m, m * p, p * n)
        T_residual = T_residual - T_rank1
        
        U_list.append(u1)
        V_list.append(v1.flatten())
        W_list.append(np.ones(p * n))  # placeholder, need to redo
        
        # This naive approach won't work because W is coupled.
        # Let me use a different approach.
        break  # This simple peeling doesn't handle the 3-way coupling correctly

    # Better approach: Use the known Strassen block construction but then
    # apply nullspace reduction on the combined factor matrix.
    
    # Build the rank-66 block construction (same as before)
    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = 1
        U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[0, 3] = -1; W[1, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = 1; W[2, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[2, 5] = 1; W[2, 5] = 1; W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1; V[1, 6] = 1; V[3, 6] = 1; W[0, 6] = 1; W[1, 6] = 1
        return U, V, W

    def get_mult_1x2_2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Build rank-66 construction
    U_66 = np.zeros((12, 66))
    V_66 = np.zeros((24, 66))
    W_66 = np.zeros((18, 66))
    
    def place_block(U_block, V_block, W_block, u_rows, v_rows, w_rows, offset):
        rank_block = U_block.shape[1]
        for c in range(rank_block):
            for local_r, global_r in enumerate(u_rows):
                U_66[global_r, offset + c] = U_block[local_r, c]
            for local_r, global_r in enumerate(v_rows):
                V_66[global_r, offset + c] = V_block[local_r, c]
            for local_r, global_r in enumerate(w_rows):
                W_66[global_r, offset + c] = W_block[local_r, c]

    S_U, S_V, S_W = get_strassen_2x2()
    M12_U, M12_V, M12_W = get_mult_1x2_2x2()
    
    def get_u_idx(i_start, i_end, j_start, j_end):
        return [i * m + j for i in range(i_start, i_end) for j in range(j_start, j_end)]
    def get_v_idx(j_start, j_end, k_start, k_end):
        return [j * p + k for j in range(j_start, j_end) for k in range(k_start, k_end)]
    def get_w_idx(k_start, k_end, i_start, i_end):
        return [k * n + i for k in range(k_start, k_end) for i in range(i_start, i_end)]

    col_offset = 0
    for j_block in range(2):
        for k_block in range(3):
            j_start = j_block * 2
            j_end = j_block * 2 + 2
            k_start = k_block * 2
            k_end = k_block * 2 + 2
            u_rows = get_u_idx(0, 2, j_start, j_end)
            v_rows = get_v_idx(j_start, j_end, k_start, k_end)
            w_rows = get_w_idx(k_start, k_end, 0, 2)
            place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
            u_rows = get_u_idx(2, 3, j_start, j_end)
            v_rows = get_v_idx(j_start, j_end, k_start, k_end)
            w_rows = get_w_idx(k_start, k_end, 2, 3)
            place_block(M12_U, M12_V, M12_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 4

    # Now apply nullspace reduction:
    # The tensor equation is: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    # If we have a valid decomposition of rank R, and there exists a vector z in R^R
    # such that the tensor sum_c z_c * (U[:,c] ⊗ V[:,c] ⊗ W[:,c]) = 0,
    # then we can eliminate one component.
    # 
    # More practically: form the "flattening" matrix. For each (i,j) pair in U's rows,
    # the contribution to T[i*m+j, :, :] is sum_c U[i*m+j, c] * V[:,c] ⊗ W[:,c]
    # 
    # A simpler check: if two columns of the "combined" factor are linearly dependent
    # in a way that preserves the tensor identity, we can merge them.
    #
    # Practical approach: Try removing each column c and see if the remaining
    # (R-1) columns can still reproduce T (with adjusted values).
    # This is equivalent to checking if column c is in the span of others
    # in the tensor sense.
    
    # Build the full tensor from the 66-rank decomposition
    T_check = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for c in range(66):
        T_check += np.outer(np.outer(U_66[:, c], V_66[:, c]), W_66[:, c]).reshape(n*m, m*p, p*n)
    
    # Verify the 66-rank works
    if np.max(np.abs(T_check - T)) < 1e-9:
        # Try nullspace reduction
        # For each column c, check if we can express it as a linear combination of others
        # such that the tensor contribution is zero.
        # 
        # The key insight: if for some c, the vectors U_66[:,c], V_66[:,c], W_66[:,c]
        # are "redundant" in the sense that U_66[:,c] = sum_{c'≠c} a_{c'} U_66[:,c']
        # AND V_66[:,c] = sum_{c'≠c} b_{c'} V_66[:,c'] AND W_66[:,c] = sum_{c'≠c} d_{c'} W_66[:,c']
        # with the product constraints... this is complex.
        #
        # Simpler: Use the fact that if we form the matrix M where M[entry, c] = 
        # the c-th factor's contribution to that tensor entry, then rank(M) gives
        # the true rank. But M is 72 x 66 and has rank at most 66.
        #
        # Actually, the most practical approach: try to find a column c such that
        # removing it and solving for a correction in the remaining columns works.
        # This is a linear system.
        
        # Alternative practical approach: SVD-based rank reduction
        # Form the "mode-1 flattening": for each i (row of U), the slice T[i, :, :]
        # = sum_c U[i,c] * (V[:,c] ⊗ W[:,c])
        # This is a linear map from R^66 to R^(m*p * p*n)
        
        # Let's try a different structural approach:
        # Use random linear combinations of columns to find dependencies.
        
        # For each pair of columns (c1, c2), check if we can find alpha, beta
        # such that the tensor contribution of alpha*col_c1 + beta*col_c2 = 0
        # (i.e., the combined rank-1 tensor is zero).
        # This requires: alpha*U_66[:,c1] ⊗ alpha*V_66[:,c1] ⊗ alpha*W_66[:,c1] +
        #                beta*U_66[:,c2] ⊗ beta*V_66[:,c2] ⊗ beta*W_66[:,c2] = 0
        # Which means the two rank-1 tensors must be negatives of each other,
        # so U_66[:,c1] = -beta/alpha * U_66[:,c2], etc.
        # This only works for exact scalar multiples, which is rare.
        
        # Better: Try to reduce by finding a column that is a linear combination
        # of others in ALL three factors simultaneously.
        
        # Most practical: just verify and return the 66-rank if it works.
        # The nullspace finding is the key new element.
        
        # Let's try: for each column c, project it out and see if the residual
        # tensor can be represented with the remaining 65 columns.
        # This means: does there exist a vector x in R^65 such that
        # sum_{c'≠c} x_{c'} * (U_66[:,c'] ⊗ V_66[:,c'] ⊗ W_66[:,c']) = T - 0
        # (since we're removing column c entirely)
        # 
        # This is a linear system! Flatten the tensor equation.
        # For each tensor entry (a, b, d): sum_{c'≠c} x_{c'} * U_66[a,c'] * V_66[b,c'] * W_66[d,c'] = T[a,b,d]
        
        reduced = False
        for c_remove in range(66):
            if time.time() > deadline - 10:
                break
            
            # Build the linear system: A x = b
            # A has shape (n*m * m*p * p*n, 65) = (72*24*18, 65) = (31104, 65)
            # b has shape (31104,)
            
            # This is too large to build explicitly in reasonable time for all 66 columns.
            # Instead, use a smaller test: pick a subset of tensor entries.
            
            # Actually, let's use a smarter check:
            # The column c is redundant iff for ALL (a,b,d):
            # T[a,b,d] = sum_{c'≠c} x_{c'} * U_66[a,c'] * V_66[b,c'] * W_66[d,c']
            # 
            # This is equivalent to: the tensor T - 0 (we remove col c's contribution)
            # must be in the span of the remaining rank-1 tensors.
            # 
            # Quick check: if U_66[:,c] is in the column space of U_66[:, [c'≠c]],
            # AND similarly for V and W, it might be reducible. But that's necessary not sufficient.
            
            # Practical approach: solve a least-squares system on a subset of entries.
            # Pick 200 random tensor entries to form a 200x65 system.
            
            remaining_cols = [c for c in range(66) if c != c_remove]
            
            # Sample tensor entries
            np.random.seed(c_remove * 1000)
            num_samples = min(200, n*m*m*p*p*n)
            idx_a = np.random.randint(0, n*m, num_samples)
            idx_b = np.random.randint(0, m*p, num_samples)
            idx_d = np.random.randint(0, p*n, num_samples)
            
            # Build A (num_samples x 65)
            A = np.zeros((num_samples, 65))
            b = np.zeros(num_samples)
            for s in range(num_samples):
                a, bb, d = idx_a[s], idx_b[s], idx_d[s]
                for r, c_r in enumerate(remaining_cols):
                    A[s, r] = U_66[a, c_r] * V_66[bb, c_r] * W_66[d, c_r]
                b[s] = T[a, bb, d]
            
            # Solve least squares
            try:
                x, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            except:
                continue
            
            # Check if the solution is good: residual should be ~0
            res = np.max(np.abs(A @ x - b))
            if res < 1e-8:
                # Verify on ALL entries
                T_reconstructed = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for r, c_r in enumerate(remaining_cols):
                    T_reconstructed += x[r] * np.outer(np.outer(U_66[:, c_r], V_66[:, c_r]), W_66[:, c_r]).reshape(n*m, m*p, p*n)
                
                if np.max(np.abs(T_reconstructed - T)) < 1e-8:
                    # Success! We found a rank-65 decomposition
                    new_U = np.column_stack([U_66[:, c_r] * x[r] for c_r in remaining_cols])
                    new_V = np.column_stack([V_66[:, c_r] * x[r] for c_r in remaining_cols])
                    new_W = np.column_stack([W_66[:, c_r] * x[r] for c_r in remaining_cols])
                    
                    # Verify
                    if verify(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                        best_U = new_U.tolist()
                        best_V = new_V.tolist()
                        best_W = new_W.tolist()
                        best_rank = 65
                        reduced = True
                        break
        
        if not reduced:
            # Use the 66-rank decomposition
            best_U = U_66.tolist()
            best_V = V_66.tolist()
            best_W = W_66.tolist()
            best_rank = 66

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
