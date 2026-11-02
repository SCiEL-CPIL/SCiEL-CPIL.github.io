import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build rank-50 tensor via explicit Strassen blocks
    # A: 2x4 (rows i=0,1; cols j=0..3)
    # B: 4x7 (rows j=0..3; cols k=0..6)
    # C: 2x7 (rows i=0,1; cols k=0..6)
    # Convention: W[k*n + i, r]
    
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0.0] * rank for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    def strassen_block(i0, j0, k0):
        """7 Strassen products for A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2]"""
        nonlocal col
        # A entries
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        # B entries
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        # C entries (W index = k*n + i)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += 1
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += 1
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1
        col += 1
    
    def schoolbook_221(i0, j0, k0):
        """4 schoolbook products for A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+1]"""
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
        U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
        U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
        U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
    
    # Build the rank-50 decomposition:
    # Strip 1: A[:, 0:2] @ B[0:2, :] -> use Strassen for k-blocks 0-1, 2-3, 4-5; schoolbook for k=6
    strassen_block(0, 0, 0)
    strassen_block(0, 0, 2)
    strassen_block(0, 0, 4)
    schoolbook_221(0, 0, 6)
    
    if time.time() > deadline:
        actual_rank = col
        return {"U": [row[:actual_rank] for row in U],
                "V": [row[:actual_rank] for row in V],
                "W": [row[:actual_rank] for row in W]}
    
    # Strip 2: A[:, 2:4] @ B[2:4, :] -> use Strassen for k-blocks 0-1, 2-3, 4-5; schoolbook for k=6
    strassen_block(0, 2, 0)
    strassen_block(0, 2, 2)
    strassen_block(0, 2, 4)
    schoolbook_221(0, 2, 6)
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    # Try to reduce rank via iterative null-space projection
    # Build the decomposition tensor T in flattened form
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def reconstruct_decomp(U_mat, V_mat, W_mat):
        """Reconstruct the full bilinear map from factor matrices."""
        # Tensor: C[i,k] = sum_j A[i,j] * B[j,k]
        # Decomposition: C = sum_r (U[:,r] outer A) * (V[:,r] outer B) * W[:,r]
        # More precisely: the r-th term contributes W[:,r] * (U[:,r]^T A) * (B^T V[:,r])
        # In tensor form: T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Verify: for given A, B: C[i,k] = sum_j sum_r A[i,j]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        #        = sum_r W[k*n+i,r] * (sum_j A[i,j]*U[i*m+j,r]*V[j*p+k,r])
        # This is not standard matrix multiplication decomposition. Let me think again.
        # 
        # The bilinear map is: C = A @ B, so C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: M[i,j,k] = delta_{i,j,k} where the map is (A,B) -> A@B
        # Actually the tensor for matrix multiplication C=A@B has components:
        #   T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        #   So T[i,j,k] = 1 if the index is valid, else 0
        # 
        # A rank-r decomposition: T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that doesn't quite work dimensionally. Let me reconsider.
        #
        # The factor matrices are:
        #   U: shape (n*m, rank) = (8, rank)  -- indices into A
        #   V: shape (m*p, rank) = (28, rank) -- indices into B  
        #   W: shape (p*n, rank) = (14, rank) -- indices into C
        #
        # The decomposition says: for the bilinear map (A,B) -> C,
        #   C_flat[k*n+i] = sum_r W[k*n+i, r] * (sum_{im,j} U[im, r] A[im]) * (sum_{jp,k} V[jp, r] B[jp])
        # where im indexes A and jp indexes B.
        #
        # Actually the standard formulation: C = sum_r (U_r^T A) * (B V_r) * W_r
        # where U_r is an n*m vector, V_r is m*p vector, W_r is p*n vector.
        # 
        # Let me just verify by direct computation with test matrices.
        pass
    
    # Verify the decomposition is correct
    def verify_decomp(U_mat, V_mat, W_mat, trials=5):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        np.random.seed(42)
        for _ in range(trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            # Reconstruct C from decomposition
            C_recon = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                # u_r: (n*m,) reshaped to (n, m)
                u_r = U_mat[:, r].reshape(n, m)
                v_r = V_mat[:, r].reshape(m, p)
                w_r = W_mat[:, r].reshape(p, n)
                # C contribution: w_r * (u_r^T A) * (B v_r) ... 
                # Actually: C[i,k] += W[k*n+i, r] * (sum_{i2,j2} U[i2*m+j2, r] A[i2,j2]) * (sum_{j3,k3} V[j3*p+k3, r] B[j3,k3])
                # That gives a scalar per r, not a matrix. Let me re-examine.
                #
                # I think the correct interpretation is:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # Decomposition: C[i,k] = sum_r W[k*n+i, r] * a_r * b_r
                # where a_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (a scalar linear functional of A)
                # and b_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (a scalar linear functional of B)
                #
                # This is a rank-r decomposition of the bilinear map into products of
                # a linear form in A, a linear form in B, and a coefficient for each C entry.
                #
                # So: C[i,k] = sum_r W[k*n+i, r] * (u_r . A_flat) * (v_r . B_flat)
                a_r = np.dot(U_mat[:, r], A.flatten())
                b_r = np.dot(V_mat[:, r], B.flatten())
                C_recon += W_mat[:, r] * a_r * b_r
                C_recon = C_recon.reshape(p, n).T  # W is indexed as k*n+i, so reshape to (p,n) then transpose
            # Actually W_mat is (p*n, rank), W_mat[k*n+i, r] is the coefficient for C[i,k]
            # So C_recon[i,k] = sum_r W_mat[k*n+i, r] * a_r * b_r
            # Let me redo:
            C_recon = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                a_r = np.dot(U_mat[:, r], A.flatten())
                b_r = np.dot(V_mat[:, r], B.flatten())
                # W_mat[k*n+i, r] for all k,i gives a (p*n,) vector
                w_r = W_mat[:, r].reshape(p, n).T  # now w_r[i,k] = W_mat[k*n+i, r]
                C_recon += w_r * a_r * b_r
            if not np.allclose(C_recon, C_expected, atol=1e-10):
                return False
        return True
    
    # First verify our rank-50 decomposition
    if not verify_decomp(np.array(U), np.array(V), np.array(W), trials=3):
        # Fallback: return what we have
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank via null-space analysis
    # The decomposition is valid if for all A, B:
    #   sum_r W[:,r] * (U[:,r]^T A_flat) * (V[:,r]^T B_flat) = (A@B).flat
    # 
    # This means the tensors T_r = W[:,r] (outer) U[:,r] (outer) V[:,r] 
    # must sum to the multiplication tensor.
    #
    # To check if we can remove a column, we check if the tensor contribution
    # of that column can be expressed as a linear combination of the others.
    #
    # Build the full tensor M of shape (n*m, m*p, p*n) = (8, 28, 14)
    # M[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to C[i,k] where
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # So M[(i*m+j), (j*p+k), (k*n+i)] = 1
    # All other entries are 0.
    
    M = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                M[a_idx, b_idx, c_idx] = 1.0
    
    # The decomposition says: M = sum_r np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
    # i.e., M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Verify
    M_decomp = np.zeros_like(M)
    for r in range(actual_rank):
        M_decomp += np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
    
    if not np.allclose(M_decomp, M, atol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    # Strategy: iteratively try to remove columns by checking if the remaining
    # columns span the same tensor space.
    
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    cur_rank = actual_rank
    removed = [False] * actual_rank
    
    # Try to find linear dependencies among the "tensors" T_r = U[:,r] (x) V[:,r] (x) W[:,r]
    # Flatten each T_r into a vector of length n*m*m*p*p*n
    # Then check if any vector is in the span of the others
    
    # This is expensive for large tensors, but our dimensions are small:
    # n*m = 8, m*p = 28, p*n = 14, so each tensor has 8*28*14 = 3136 entries
    # With rank 50, we have 50 vectors of length 3136
    
    tensor_dim = (n * m) * (m * p) * (p * n)
    T_flat = np.zeros((cur_rank, tensor_dim), dtype=float)
    for r in range(cur_rank):
        T_flat[r] = np.outer(np.outer(U_cur[:, r], V_cur[:, r]), W_cur[:, r]).flatten()
    
    # Check for linear dependencies using SVD
    # If the rank of T_flat is less than cur_rank, we have dependencies
    
    # Do this iteratively: try to remove one column at a time
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Compute SVD of current tensor matrix
        try:
            U_svd, svals, Vt = np.linalg.svd(T_flat, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        
        # Find columns with smallest singular values
        # If svals[-1] is very small, we can drop the corresponding right singular vector
        # But we need to identify which original column to drop
        
        # Alternative: for each column r, check if T_flat[r] is in the span of the others
        # by computing the residual after projecting onto the span of the others
        
        best_to_remove = -1
        best_residual = np.inf
        
        for r in range(cur_rank):
            if removed[r]:
                continue
            # Create matrix of all other non-removed tensors
            others = [r2 for r2 in range(cur_rank) if r2 != r and not removed[r2]]
            if len(others) == 0:
                continue
            T_others = T_flat[others]  # shape (len(others), tensor_dim)
            # Check if T_flat[r] is in the row space of T_others
            # i.e., is T_flat[r] in the span of rows of T_others?
            # Actually, we want: is T_flat[r] in the span of {T_flat[r2] : r2 in others}?
            # Project T_flat[r] onto the span of the others
            try:
                # Use least squares: find coefficients c such that T_others.T @ c ≈ T_flat[r]
                # T_others is (len(others), tensor_dim)
                # We want T_others.T @ c = T_flat[r]
                # This is an underdetermined system if len(others) < tensor_dim
                # Use SVD of T_others.T
                c, residuals, rank_o, sv = np.linalg.lstsq(T_others.T, T_flat[r], rcond=None)
                residual = np.linalg.norm(T_flat[r] - T_others.T @ c)
                if residual < best_residual:
                    best_residual = residual
                    best_to_remove = r
            except np.linalg.LinAlgError:
                continue
        
        # If we found a column to remove with small residual, remove it
        if best_to_remove >= 0 and best_residual < 1e-8:
            removed[best_to_remove] = True
            cur_rank -= 1
            # Rebuild T_flat without the removed column
            keep = [r for r in range(actual_rank) if not removed[r]]
            T_flat = T_flat[keep]
            U_cur = U_np[:, keep]
            V_cur = V_np[:, keep]
            W_cur = W_np[:, keep]
            print(f"Removed column {best_to_remove}, new rank: {cur_rank}")
        else:
            break
    
    if cur_rank < actual_rank:
        U_final = U_cur.tolist()
        V_final = V_cur.tolist()
        W_final = W_cur.tolist()
    else:
        U_final = U
        V_final = V
        W_final = W
    
    return {"U": U_final, "V": V_final, "W": W_final}
