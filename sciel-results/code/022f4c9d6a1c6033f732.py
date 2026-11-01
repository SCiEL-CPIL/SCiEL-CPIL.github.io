import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # A is 2x2, B is 2x10, C is 2x10
    # U shape (4, R), V shape (20, R), W shape (20, R)
    
    def build_strassen_rank_35():
        U = np.zeros((4, 35), dtype=float)
        V = np.zeros((20, 35), dtype=float)
        W = np.zeros((20, 35), dtype=float)
        
        # Standard Strassen terms for 2x2x2 tensor
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = 0
        for q in range(5):
            for a_vec, b_vec, c_vec in terms:
                U[0, col] = a_vec[0]
                U[1, col] = a_vec[1]
                U[2, col] = a_vec[2]
                U[3, col] = a_vec[3]
                
                V[0*10 + 2*q, col] = b_vec[0]
                V[0*10 + 2*q+1, col] = b_vec[1]
                V[1*10 + 2*q, col] = b_vec[2]
                V[1*10 + 2*q+1, col] = b_vec[3]
                
                W[4*q + 0, col] = c_vec[0]
                W[4*q + 1, col] = c_vec[1]
                W[4*q + 2, col] = c_vec[2]
                W[4*q + 3, col] = c_vec[3]
                
                col += 1
        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Fast vectorized verification of the tensor decomposition."""
        n, m, p = 2, 2, 10
        R = U.shape[1]
        
        # Build the full multiplication tensor T of shape (n, m, p)
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Then verify: for random A (n,m) and B (m,p), C = A@B == sum_j T[i,j,k]*A[i,j]*B[j,k]
        
        # More efficient: build T directly
        T = np.zeros((n, m, p))
        for r in range(R):
            u_col = U[:, r]  # shape (n*m,)
            v_col = V[:, r]  # shape (m*p,)
            w_col = W[:, r]  # shape (p*n,)
            
            # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # Use einsum for efficiency
            T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        
        # Now verify: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        np.random.seed(42)
        for trial in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # C_computed[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
            C_computed = np.einsum('ijk,ij,jk->ik', T, A, B)
            
            if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
                return False
        return True

    def try_rank_reduction(U, V, W, time_limit):
        """
        Try to reduce rank by checking if any column of the 35-term decomposition
        is a linear combination of the others in the tensor sense.
        
        The tensor for column r is T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
        (with appropriate index mappings).
        
        If T_r = sum_{s != r} c_s * T_s, then we can eliminate column r.
        This is a linear system in the c_s's.
        """
        n, m, p = 2, 2, 10
        R = U.shape[1]
        
        # Build all rank-1 tensors as vectors in R^(n*m*p) = R^40
        tensors = []
        for r in range(R):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # T_r[i,j,k] = u_col[i,j] * v_col[j,k] * w_col[k,i]
            T_r = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
            tensors.append(T_r.flatten())
        
        tensors = np.array(tensors)  # shape (R, n*m*p)
        
        # Check if any row is in the span of the others
        # For each r, check if tensors[r] is in span of tensors without r
        # This is equivalent to checking if the rank of the full matrix equals
        # the rank of the matrix with row r removed.
        
        # More efficiently: compute the null space of the full matrix.
        # If there's a non-trivial null vector, we can use it to eliminate a term.
        
        # Actually, the rank of the tensor decomposition is the minimum number
        # of rank-1 tensors needed. If the 35 vectors are linearly dependent,
        # we can reduce the rank.
        
        # Compute rank of the 35x40 matrix
        rank_full = np.linalg.matrix_rank(tensors, tol=1e-9)
        
        if rank_full < R:
            # There's a linear dependency. Find it.
            # Use SVD to find the null space
            U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=False)
            # Null space vectors are the last (R - rank_full) columns of Vt
            null_dim = R - rank_full
            null_vecs = Vt[rank_full:, :]  # shape (null_dim, R)
            
            # For each null vector, find the index with largest magnitude
            # and try to eliminate that column
            for nv in null_vecs:
                # Find the index with max |nv[i]|
                max_idx = np.argmax(np.abs(nv))
                if abs(nv[max_idx]) < 1e-9:
                    continue
                
                # We can express tensor[max_idx] as a combination of others:
                # tensors[max_idx] = -sum_{i != max_idx} (nv[i]/nv[max_idx]) * tensors[i]
                # So we can eliminate column max_idx
                # New rank = R - 1
                
                # Build new matrices without column max_idx
                keep = [i for i in range(R) if i != max_idx]
                new_U = U[:, keep]
                new_V = V[:, keep]
                new_W = W[:, keep]
                
                # Verify
                if verify_decomposition(new_U, new_V, new_W):
                    return new_U, new_V, new_W, R - 1
        
        return None, None, None, R

    # Build the rank-35 solution
    U, V, W = build_strassen_rank_35()
    
    # Verify correctness
    is_valid = verify_decomposition(U, V, W)
    
    best_U, best_V, best_W = U, V, W
    best_rank = 35
    
    if is_valid:
        # Try to reduce rank
        elapsed = time.time() - start_time
        if elapsed < 80:
            new_U, new_V, new_W, new_rank = try_rank_reduction(U, V, W, 80)
            if new_U is not None and new_rank < best_rank:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
