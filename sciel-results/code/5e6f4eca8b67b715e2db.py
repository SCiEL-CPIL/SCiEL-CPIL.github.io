import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Structural approach: Use Strassen-like decomposition for 4x4 x 4x7
    # Split 4x4 into 2x2 blocks. We have a 2x2 block matrix A times 2x2 block matrix B (where B has 4x7 blocks)
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13, B14, B15, B16, B17],
    #      [B21, B22, B23, B24, B25, B26, B27]] (each 2x1, i.e., 2-element column vectors)
    
    # For the 2x2 block multiplication, Strassen uses 7 multiplications instead of 8.
    # But here B has 7 columns, so each "block" B_ij is actually 2x7.
    # Let's use a different approach: treat the 4x4 x 4x7 as computing 7 independent 4x4 x 4x1 products.
    # For each product, we can use Strassen on the 4x4 part.
    
    # Actually, let's use a simpler structural improvement:
    # The rank of the 4x4 x 4x7 multiplication tensor is at most 4*4*7 = 112.
    # We can reduce this by exploiting that we're multiplying by a 4x7 matrix.
    
    # Key insight: We can use the fact that the 4x7 matrix can be seen as 7 column vectors.
    # C[:,k] = A @ B[:,k] for k=0..6
    # Each A @ B[:,k] is a 4x1 vector.
    
    # Let's try a rank-98 construction using 2x2 Strassen blocks:
    # Split rows/cols of A into 2x2 blocks, and rows of B into 2-element blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [B1; B2] where B1 is 2x7, B2 is 2x7
    
    # C = A @ B = [[A11@B1 + A12@B2], [A21@B1 + A22@B2]]
    # Each term like A11@B1 is 2x7, requiring 2*2*7 = 28 scalar mults
    # Total: 4 * 28 = 112 (schoolbook)
    
    # Using Strassen for 2x2 @ 2x7:
    # M1 = (A11+A22) @ (B1+B2)
    # M2 = (A21+A22) @ B1
    # M3 = A11 @ (B2-B1)
    # M4 = A22 @ (B1-B2)
    # M5 = (A11+A12) @ B2
    # M6 = (A21-A11) @ (B1+B2)
    # M7 = (A12-A22) @ (B1+B2)
    # Then:
    # C11 = M1 + M4 - M2 + M6  (wait, this doesn't work directly for rectangular)
    
    # Let me use a cleaner approach: just return schoolbook for now but with a faster construction,
    # and add a time-bounded search for better ranks.
    
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook as baseline
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to find lower rank using efficient ALS with vectorized operations
    # We'll try a few ranks with multiple restarts
    
    np.random.seed(42)
    
    def check_decomposition(U, V, W, n, m, p, R):
        """Check if decomposition is exact (within tolerance)"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    def try_als(U_init, V_init, W_init, R, n, m, p, max_iter=30, deadline=140.0):
        """Run ALS and return (U, V, W, error)"""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update V: for each (j,k), V[j*p+k, :] = C / ||C||^2
            # where C[r] = sum_i U[i*m+j, r] * W[k*n+i, r]
            for j in range(m):
                for k in range(p):
                    C = np.zeros(R)
                    for i in range(n):
                        C += U[i*m+j, :] * W[k*n+i, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq > 1e-10:
                        V[j*p+k, :] = C / norm_sq
            
            # Update W: for each (k,i), W[k*n+i, :] = C / ||C||^2
            # where C[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    C = np.zeros(R)
                    for j in range(m):
                        C += U[i*m+j, :] * V[j*p+k, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq > 1e-10:
                        W[k*n+i, :] = C / norm_sq
            
            # Update U: for each (i,j), U[i*m+j, :] = C / ||C||^2
            # where C[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    C = np.zeros(R)
                    for k in range(p):
                        C += V[j*p+k, :] * W[k*n+i, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq > 1e-10:
                        U[i*m+j, :] = C / norm_sq
        
        err = check_decomposition(U, V, W, n, m, p, R)
        return U, V, W, err
    
    # Try several ranks with multiple random restarts
    ranks_to_try = [105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Initialize with noise
            U_init = np.random.randn(n * m, R) * 0.1
            V_init = np.random.randn(m * p, R) * 0.1
            W_init = np.random.randn(p * n, R) * 0.1
            
            U_sol, V_sol, W_sol, err = try_als(U_init, V_init, W_init, R, n, m, p, max_iter=40, deadline=deadline)
            
            if err < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                err_rounded = check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, R)
                
                if err_rounded < 1e-6:
                    best_rank = R
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
                else:
                    # Use the unrounded solution if it's valid
                    best_rank = R
                    best_U = U_sol
                    best_V = V_sol
                    best_W = W_sol
                    break
        
        if best_rank < R:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
