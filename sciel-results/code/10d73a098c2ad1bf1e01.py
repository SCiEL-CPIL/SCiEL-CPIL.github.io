"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Build the target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # T is a 3-tensor of shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Reshape to matrix form: M[i, j*p + k] = T[i, j, k]
    # Or use the convention: T_mat[a, b] where a = i*m + j, b = k*p + ... 
    # Actually, let's use the standard flattening:
    # T_vec[i*m*j + ... ] - let's be careful
    
    # The bilinear form: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has T[i,j,k] = 1
    # Rank-1 decomposition: T = sum_r U_col_r ⊗ V_col_r ⊗ W_col_r
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # and T_mat[a, b] = sum_r U[a,r] * V[b,r] * W[c,r] ... 
    
    # Wait, let me reconsider. The standard formulation:
    # We want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]? No.
    
    # Actually, for matrix multiplication (n×m)(m×p) = (n×p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that C[i,k] = sum_{j,r} A[i,j] T[i,j,k,r] B[j,k]
    # No, that's not right either.
    
    # The correct formulation: The multiplication tensor T ∈ R^{n×m × m×p × n×p}
    # where T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0.
    # Wait no, T[a,b,c] should be such that sum_{a,b} T[a,b,c] A[a] B[b] = C[c]
    
    # Let me use the convention from the problem:
    # U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    # W[k*n + i, r] is the coefficient for output C[i,k] from term r
    # The decomposition is: for each r, the term is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # And the sum over r should give the multiplication tensor.
    
    # The multiplication tensor T has T[a,b,c] = 1 if there exist i,j,k with:
    # a = i*m + j, b = j*p + k, c = k*n + i, and the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    
    # Build T as a matrix: T_mat[a, b] for fixed c? No, it's a 3-tensor.
    # Let's flatten: T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    
    T_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W[k*n + i] per the convention
                T_tensor[a, b, c] = 1
    
    # We want to find rank-1 tensors U_r, V_r, W_r such that:
    # T_tensor[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # and their sum equals T_tensor.
    
    # To solve this, we can fix two of the matrices and solve for the third.
    # Let's try ranks from low to high.
    
    start_time = time.time()
    deadline = 130  # Leave 20s margin
    
    def try_rank(r, max_iters=100, tol=1e-8):
        """Try to find a rank-r decomposition using ALS."""
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: fix V, W. For each a, solve for U[a,:]
            # T[a,:,c] = sum_r U[a,r] V[:,r] ⊗ W[c,r] (as a vector in b)
            # More precisely: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T[a, b, c] over b,c = sum_r U[a,r] (V[:,r] ⊗ W[:,r])[b,c]
            # So T_a (matrix of size m*p x p*n) = U[a,:] @ (V ⊙ W) where (V⊙W)[r] = V[:,r] ⊗ W[:,r]
            
            # Build the basis matrix for this update
            # B_u has shape (m*p * p*n, r) where B_u[b*(p*n) + c, r] = V[b,r] * W[c,r]
            B_u = np.zeros((m*p * p*n, r))
            for r_idx in range(r):
                B_u[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
            
            # T_u has shape (n*m, m*p * p*n) where T_u[a, b*(p*n)+c] = T[a,b,c]
            T_u = T_tensor.reshape(n*m, m*p * p*n)
            
            # Solve: T_u[a,:] = U[a,:] @ B_u^T
            # U[a,:] = T_u[a,:] @ pinv(B_u^T)
            B_u_T = B_u.T  # shape (r, m*p*p*n)
            pinv_Bu = np.linalg.pinv(B_u_T)  # shape (m*p*p*n, r)
            U_new = T_u @ pinv_Bu  # shape (n*m, r)
            
            # Update V
            B_v = np.zeros((n*m * p*n, r))
            for r_idx in range(r):
                B_v[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
            
            T_v = T_tensor.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
            
            B_v_T = B_v.T
            pinv_Bv = np.linalg.pinv(B_v_T)
            V_new = T_v @ pinv_Bv
            
            # Update W
            B_w = np.zeros((n*m * m*p, r))
            for r_idx in range(r):
                B_w[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
            
            T_w = T_tensor.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
            
            B_w_T = B_w.T
            pinv_Bw = np.linalg.pinv(B_w_T)
            W_new = T_w @ pinv_Bw
            
            # Check convergence
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            T_approx = np.zeros((n*m, m*p, p*n))
            for r_idx in range(r):
                T_approx += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
            
            error = np.linalg.norm(T_tensor - T_approx) / np.linalg.norm(T_tensor)
            
            if error < tol:
                return U, V, W, error
        
        return None
    
    # Try ranks from 20 up to 56
    best_result = None
    for r in range(20, 57):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=50)
        if result is not None:
            U, V, W, error = result
            best_result = (r, U, V, W, error)
            break
    
    # If we found a solution, return it
    if best_result is not None:
        r, U, V, W, error = best_result
        
        # Round to clean values if possible
        # Check if entries are close to integers or half-integers
        def round_matrix(M):
            # Try rounding to nearest 0.5
            M_rounded = np.round(M * 2) / 2
            if np.allclose(M, M_rounded, atol=1e-6):
                return M_rounded
            # Try rounding to integers
            M_int = np.round(M)
            if np.allclose(M, M_int, atol=1e-6):
                return M_int
            return M
        
        U = round_matrix(U)
        V = round_matrix(V)
        W = round_matrix(W)
        
        # Verify
        T_approx = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_approx += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
        
        if not np.allclose(T_tensor, T_approx, atol=1e-6):
            # Fall back to schoolbook
            pass
        else:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook decomposition
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
