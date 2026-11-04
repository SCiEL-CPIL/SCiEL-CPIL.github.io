"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decomposition
    # The tensor T[i,j,k] = delta for the multiplication
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the full tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # Each entry is 1 in the "basis" sense
    
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each term A[i,j]*B[j,k] contributes
    
    # Let's try rank r by solving for U, V, W
    # This is a non-convex problem, so let's try small ranks and use random initialization
    
    best_result = (U_school, V_school, W_school, rank_schoolbook)
    
    # Try ranks from some lower bound up to schoolbook
    # Lower bound for M_{n,m,p} is max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But actual rank is likely higher. Let's try from 20 to 44
    
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    for r in range(20, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
        
        # Try to find a decomposition of rank r
        # Use random initialization and iterative refinement
        for attempt in range(5):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(r * 100 + attempt)
            
            # Initialize U, V, W randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Iteratively solve: given V and W, solve for U, etc.
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Reshape for solving
            # For fixed (j,k): sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k] for all i
            # This is: U_row * diag(V_row * W_col) = T_slice
            # More precisely: for each (j,k), we have a vector equation in i:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            
            # Let's do alternating least squares
            converged = False
            for it in range(100):
                if time.time() - start_time > deadline:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] for all k
                # Let A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then U[i*m+j, :] should satisfy: A_ij @ U[i*m+j, :] = T[i,j,:]
                
                U_new = U.copy()
                for i in range(n):
                    for j in range(m):
                        # Build A matrix: (p, r)
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve A_mat @ U[i*m+j, :] = T[i,j,:]
                        try:
                            U_new[i * m + j, :] = np.linalg.lstsq(A_mat, T[i, j, :], rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] for all i
                V_new = V.copy()
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            V_new[j * p + k, :] = np.linalg.lstsq(A_mat, T[:, j, k], rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] for all j
                W_new = W.copy()
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            W_new[k * n + i, :] = np.linalg.lstsq(A_mat, T[i, :, k], rcond=None)[0]
                        except:
                            pass
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            error += (val - T[i, j, k]) ** 2
                error = np.sqrt(error)
                
                if error < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = (U_list, V_list, W_list, r)
                break  # Found rank r, try to go lower? No, we're going up, so this is good
    
    U_out, V_out, W_out, r_out = best_result
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
