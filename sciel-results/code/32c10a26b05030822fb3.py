"""Reconstructed approach: use explicit known decompositions via block structure."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Convention: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    For 2x2: n=m=p=2, so U is (4,7), V is (4,7), W is (4,7)
    """
    n = m = p = 2
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i*m+j, t] gives coefficient of A[i,j] in the t-th left factor
    # V[j*p+k, t] gives coefficient of B[j,k] in the t-th right factor
    # W[k*n+i, t] gives coefficient in C[i,k] from the t-th product
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Index: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i (trace convention: W[k*n+i])
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1  # A00, A11
    V[0,0] = 1; V[3,0] = 1  # B00, B11
    W[0,0] = 1; W[3,0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1  # A10, A11
    V[0,1] = 1              # B00
    W[2,1] = 1; W[3,1] = -1 # C10, C11 (C21=C10 in 0-indexed)
    
    # M3 = A00(B10-B11)
    U[0,2] = 1              # A00
    V[2,2] = 1; V[3,2] = -1 # B10, B11
    W[1,2] = 1; W[2,2] = 1  # C01, C10
    
    # M4 = A11(B01-B00)
    U[3,3] = 1              # A11
    V[1,3] = 1; V[0,3] = -1 # B01, B00
    W[0,3] = 1; W[2,3] = 1  # C00, C10
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1  # A00, A01
    V[3,4] = 1              # B11
    W[1,4] = 1; W[3,4] = -1 # C01, C11
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1 # A10, A00
    V[0,5] = 1; V[1,5] = 1  # B00, B01
    W[0,5] = 1; W[2,5] = 1  # C00, C10
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1 # A01, A11
    V[2,6] = 1; V[3,6] = 1  # B10, B11
    W[1,6] = 1; W[3,6] = 1  # C01, C11
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication.
    For all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should give
    the correct contribution. Actually, we need to check that the bilinear map
    is correctly represented.
    
    The condition is: for all i,j,k, the sum over t of U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    should equal 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which is always
    the case for standard matrix mult. But wait, that's the all-ones condition again.
    
    Actually, I think the correct verification is: the decomposition should satisfy
    that when we compute C[i,k] = sum_j A[i,j]*B[j,k], the tensor equation holds.
    
    Let me just verify numerically with random matrices.
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using the decomposition
    # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_k' V[j'*p+k', t] * B[j',k'])
    # Wait, that's not quite right. Let me think...
    
    # Actually, in a rank-r decomposition of the matrix multiplication tensor:
    # The r scalar multiplications are: M_t = (left_t) * (right_t)
    # where left_t = sum_j U[i*m+j, t] * A[i,j] for each i
    # and right_t = sum_k V[j*p+k, t] * B[j,k] for each k
    # Then C[i,k] = sum_t W[k*n+i, t] * M_t
    
    # But this doesn't quite work because left_t depends on i and right_t depends on k.
    
    # I think the correct interpretation is:
    # For each term t, we compute:
    #   L_t = sum_{i,j} U[i*m+j, t] * A[i,j]  (a scalar)
    #   R_t = sum_{j,k} V[j*p+k, t] * B[j,k]  (a scalar)  
    #   M_t = L_t * R_t
    # Then C[i,k] = sum_t W[k*n+i, t] * M_t
    
    # Let's verify this interpretation.
    
    C_computed = np.zeros((n, p))
    for t in range(U.shape[1]):
        L_t = np.sum(U[:, t] * A.flatten())
        R_t = np.sum(V[:, t] * B.flatten())
        M_t = L_t * R_t
        C_computed += W[:, t] * M_t
    
    # Reshape C_computed: W has shape (p*n, rank), so W[k*n+i, t]
    # C_computed is (p*n,) flattened, need to reshape to (n, p) with C[i,k] = C_computed[k*n+i]
    C_computed = C_computed.reshape((p, n)).T  # C_computed[k*n+i] -> C[i,k]
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
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
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try block decomposition: split 5 into 2+3
    # Use Strassen for 2x2 (rank 7) and a decomposition for 3x3
    # The 3x3 matrix multiplication has known rank <= 23 (but constructing is complex)
    # For now, let's try to verify if we can use a simpler approach
    
    # Actually, let me try a different approach: use the fact that we can verify
    # decompositions numerically, and search for low-rank decompositions.
    
    # Strategy: Start with a random sparse initialization and use iterative refinement
    # to find a valid decomposition with lower rank.
    
    # But first, let's try to construct an explicit decomposition using block structure.
    
    # For 5x5, we can use a 2x2 block of 2x2 matrices and a 3x3 block.
    # However, the cross terms make this complex.
    
    # Alternative: Use a known result. The rank of 5x5 matrix multiplication is
    # known to be at most 49 (from various constructions). Let me try to construct
    # one using a combination of Strassen blocks.
    
    # Actually, let me try a simpler approach: use random initialization and 
    # gradient-based optimization to find a valid decomposition.
    
    # But given the complexity, let me just try to verify the schoolbook and 
    # attempt a few structured improvements.
    
    # Verify schoolbook
    if not _verify(U_sb, V_sb, W_sb, n, m, p):
        # This shouldn't happen, but handle it
        pass
    
    # Try to find a lower rank decomposition using a structured search
    # For 5x5, try ranks from 100 down to 49
    
    # Use a random sparse approach: initialize with random ±1 entries and 
    # try to satisfy the tensor equation
    
    # The tensor equation is: for all i,j,k:
    # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    # This is a system of n*m*p = 125 equations in r*3 variables (where r is rank).
    # For r < 125, this is underdetermined, so solutions should exist.
    
    # Let's try to solve this using a random linear algebra approach.
    
    # Actually, the equation is trilinear, not linear. So it's a system of 
    # polynomial equations. This is hard to solve directly.
    
    # Let me try a different approach: use the Strassen decomposition for 2x2 
    # and pad it to 5x5.
    
    # Create a 5x5 decomposition by placing the 2x2 Strassen decomposition
    # in the top-left corner and using schoolbook for the rest.
    
    # This won't be optimal, but it should give rank < 125.
    
    # Actually, let me just try a numerical optimization approach.
    
    # Initialize with random values and use alternating least squares
    # to minimize the residual of the tensor equation.
    
    np.random.seed(12345)
    
    # Try several rank values
    for target_rank in [100, 90, 80, 70, 60, 55, 50, 49]:
        if time.time() > deadline - 20:
            break
        
        # Initialize randomly
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Use alternating minimization to satisfy the tensor equation
        # For each t, the contribution to the tensor is U[:,t] ⊗ V[:,t] ⊗ W[:,t]
        # We want sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T where T[i,j,k] = 1
        
        # This is equivalent to: for all i,j,k:
        # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        
        # Let's use a simple iterative approach: fix two matrices and solve for the third.
        
        max_iter = 200
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Fix U and V, solve for W
            # For each (i,k): sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] * V[j*p+k,t]) = 1
            # Let S[i,k,t] = sum_j U[i*m+j,t] * V[j*p+k,t]
            # Then: sum_t W[k*n+i,t] * S[i,k,t] = 1 for all i,k
            
            # This is a linear system: W @ S^T = ones(n,p) where W is (p*n, rank)
            # Reshape: W_flat (p*n, rank), S (n, p, rank)
            
            S = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        S[i, k, :] += U[i * m + j, :] * V[j * p + k, :]
            
            # Now solve: for each (i,k), sum_t W[k*n+i, t] * S[i,k,t] = 1
            # This is: W_row @ S[i,k,:] = 1 where W_row = W[k*n+i, :]
            
            # Stack all equations: A @ W_flat = b
            # A has shape (n*p, rank) where A[i*p+k, t] = S[i,k,t]
            # b has shape (n*p,) with all 1s
            # W_flat has shape (p*n, rank)
            
            A_mat = S.reshape(n * p, target_rank)
            b_vec = np.ones(n * p)
            
            # Solve A_mat @ W_col = b_vec for each column of W
            # W_col has shape (rank, rank)? No, W is (p*n, rank)
            # We need: for each t, W[:, t] @ A_mat.T = b_vec? No...
            
            # Actually: sum_t W[k*n+i, t] * S[i,k,t] = 1
            # Let W_ik = W[k*n+i, :] (a row vector of length target_rank)
            # Then W_ik @ S[i,k,:] = 1
            
            # This is a system of n*p equations in p*n*target_rank unknowns.
            # We can solve it by setting up the full linear system.
            
            # Let me reshape differently. Let W_mat = W (shape p*n, rank)
            # The equation is: for all i,k: sum_t W_mat[k*n+i, t] * S[i,k,t] = 1
            
            # This can be written as: trace(W_mat^T @ S_reshaped) = n*p
            # where S_reshaped has appropriate shape.
            
            # Alternatively, vectorize: 
            # Let w = W_mat.flatten() (shape p*n*rank)
            # Let s = S.flatten() (shape n*p*rank)
            # The equation is a set of n*p bilinear forms.
            
            # This is getting complex. Let me use a simpler approach:
            # Use numpy's lstsq to solve the underdetermined system.
            
            # Set up the system: for each (i,k), we have one equation
            # sum_t W[k*n+i, t] * S[i,k,t] = 1
            
            # This is: M @ W_flat = 1 where M is (n*p, p*n*rank)
            # M[i*p+k, (k*n+i)*rank + t] = S[i,k,t]
            
            M = np.zeros((n * p, p * n * target_rank))
            for i in range(n):
                for k in range(p):
                    for t in range(target_rank):
                        row = i * p + k
                        col = (k * n + i) * target_rank + t
                        M[row, col] = S[i, k, t]
            
            W_flat, residuals, rank_m, sv = np.linalg.lstsq(M, b_vec, rcond=None)
            W_mat = W_flat.reshape(p * n, target_rank)
            
            # Check residual
            residual_W = np.linalg.norm(M @ W_flat - b_vec)
            
            if residual_W < 1e-8:
                # Now fix W and V, solve for U
                # For each (i,j): sum_t U[i*m+j, t] * (sum_k V[j*p+k, t] * W[k*n+i, t]) = 1
                # Let T[i,j,t] = sum_k V[j*p+k, t] * W[k*n+i, t]
                # Then: sum_t U[i*m+j, t] * T[i,j,t] = 1 for all i,j
                
                T = np.zeros((n, m, target_rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, :] += V[j * p + k, :] * W[k * n + i, :]
                
                A_mat2 = T.reshape(n * m, target_rank)
                b_vec2 = np.ones(n * m)
                
                M2 = np.zeros((n * m, n * m * target_rank))
                for i in range(n):
                    for j in range(m):
                        for t in range(target_rank):
                            row = i * m + j
                            col = (i * m + j) * target_rank + t
                            M2[row, col] = T[i, j, t]
                
                U_flat, _, _, _ = np.linalg.lstsq(M2, b_vec2, rcond=None)
                U_mat = U_flat.reshape(n * m, target_rank)
                
                residual_U = np.linalg.norm(M2 @ U_flat - b_vec2)
                
                if residual_U < 1e-8:
                    # Now fix U and W, solve for V
                    # For each (j,k): sum_t V[j*p+k, t] * (sum_i U[i*m+j, t] * W[k*n+i, t]) = 1
                    # Let R[j,k,t] = sum_i U[i*m+j, t] * W[k*n+i, t]
                    # Then: sum_t V[j*p+k, t] * R[j,k,t] = 1 for all j,k
                    
                    R = np.zeros((m, p, target_rank))
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                R[j, k, :] += U_mat[i * m + j, :] * W_mat[k * n + i, :]
                    
                    A_mat3 = R.reshape(m * p, target_rank)
                    b_vec3 = np.ones(m * p)
                    
                    M3 = np.zeros((m * p, m * p * target_rank))
                    for j in range(m):
                        for k in range(p):
                            for t in range(target_rank):
                                row = j * p + k
                                col = (j * p + k) * target_rank + t
                                M3[row, col] = R[j, k, t]
                    
                    V_flat, _, _, _ = np.linalg.lstsq(M3, b_vec3, rcond=None)
                    V_mat = V_flat.reshape(m * p, target_rank)
                    
                    residual_V = np.linalg.norm(M3 @ V_flat - b_vec3)
                    
                    if residual_V < 1e-8:
                        # All three solved, verify
                        if _verify(U_mat, V_mat, W_mat, n, m, p):
                            if target_rank < best_rank:
                                best_rank = target_rank
                                best_U, best_V, best_W = U_mat, V_mat, W_mat
                            break  # Found valid decomposition at this rank, try lower
            
        if time.time() > deadline - 10:
            break
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END

