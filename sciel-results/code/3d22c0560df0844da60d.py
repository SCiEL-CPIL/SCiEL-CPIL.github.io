"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    U_rows = n * m  # 12
    V_rows = m * p  # 20
    W_rows = p * n  # 15
    
    deadline = time.time() + 110  # safety margin
    
    # Build the tensor T of size (n*m) x (m*p) x (p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a, b, c] = 0
    T = np.zeros((U_rows, V_rows, W_rows), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Start with schoolbook decomposition (rank = n*m*p = 60)
    rank = n * m * p
    U = np.zeros((U_rows, rank), dtype=np.float64)
    V = np.zeros((V_rows, rank), dtype=np.float64)
    W = np.zeros((W_rows, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=500):
        """Try to reduce the rank by removing redundant columns or combining them."""
        current_rank = U_mat.shape[1]
        
        # Try to find pairs of columns that can be combined
        # For each pair (r1, r2), check if we can find coefficients alpha, beta
        # such that the combined column still satisfies the tensor equation
        # This is a complex optimization problem, so we use a simpler approach:
        # Try to zero out one column by expressing it as a linear combination of others
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
                
            if current_rank <= 1:
                break
            
            # Try to remove the last column by expressing it as a combination of others
            # This is a least squares problem: find x such that U[:, -1] ≈ U[:, :-1] @ x
            # But we need to satisfy the full tensor constraint, which is more complex
            
            # Simpler approach: try random linear combinations
            for _ in range(10):
                r1 = np.random.randint(0, current_rank - 1)
                r2 = current_rank - 1
                
                # Try to merge r2 into r1 by scaling
                # We need to find alpha, beta such that:
                # alpha * U[:, r1] + beta * U[:, r2] works for all constraints
                # This is hard to solve directly, so we try a heuristic
                
                # Try setting beta = 0 and see if we can just remove r2
                # Check if column r2 is a linear combination of others
                if current_rank > 2:
                    A = np.column_stack([U_mat[:, :r2], U_mat[:, r2+1:current_rank]])
                    b_vec = U_mat[:, r2]
                    
                    # Check if b_vec is in the span of A
                    try:
                        # Use least squares to find coefficients
                        coeffs, residuals, rank_a, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                        reconstruction = A @ coeffs
                        if np.linalg.norm(reconstruction - b_vec) < 1e-6:
                            # Column r2 is redundant, remove it
                            mask = np.ones(current_rank, dtype=bool)
                            mask[r2] = False
                            U_new = U_mat[:, mask]
                            V_new = V_mat[:, mask]
                            W_new = W_mat[:, mask]
                            
                            if verify(U_new, V_new, W_new):
                                U_mat = U_new
                                V_mat = V_new
                                W_mat = W_new
                                current_rank -= 1
                                break
                    except:
                        pass
            
            if time.time() > deadline:
                break
        
        return U_mat, V_mat, W_mat, current_rank
    
    # First, verify the schoolbook decomposition
    if not verify(U, V, W):
        # Should not happen, but just in case
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    U, V, W, rank = try_reduce_rank(U, V, W)
    
    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
