"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Start with schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower-rank decompositions using random initialization + iterative refinement
    # Strategy: try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # Known results suggest ranks around 30-40 might be achievable
    
    lower_bound = max(n * m, m * p, p * n)
    
    # Try various ranks in descending order from schoolbook down to lower_bound
    # Use random Gaussian initialization and check if we can solve the linear system
    
    # For a given rank r, we need:
    # U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,j,k] for all i,j,k
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns (with bilinear structure)
    
    # Approach: fix U and V, solve for W (linear). Then iterate.
    # Or: use alternating least squares / gradient descent on the tensor reconstruction error
    
    # Better approach: use the fact that for fixed rank r, we can set up a linear system
    # by fixing two matrices and solving for the third.
    
    # Let's try: for each rank r, try multiple random initializations
    # 1. Initialize U, V, W randomly
    # 2. Iterate: fix U,V, solve for W (linear least squares)
    #    fix V,W, solve for U (linear least squares)  
    #    fix W,U, solve for V (linear least squares)
    # 3. Check if the reconstruction is exact (within tolerance)
    
    def try_rank(r, num_trials=5, max_iter=200):
        """Try to find a rank-r decomposition using alternating least squares."""
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U_t = np.random.randn(n * m, r) * 0.1
            V_t = np.random.randn(m * p, r) * 0.1
            W_t = np.random.randn(p * n, r) * 0.1
            
            converged = False
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Fix U, V: solve for W
                # For each (k, i): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta
                # This is nonlinear in W when U,V are fixed... wait, no.
                # Actually for fixed U and V, the equation for W[k*n+i, c] is:
                # sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,j,k]
                # This couples different c's through the sum over j.
                
                # Alternative: use gradient-based approach
                # Compute gradient of ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2
                
                # Let's use a simpler approach: fix W, solve for U and V jointly
                # Given W, the constraint becomes:
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta[i,j,k]
                # This is still bilinear in U,V.
                
                # Let's use a different strategy: 
                # Fix V and W, solve for U (linear)
                # Fix U and W, solve for V (linear)
                # Fix U and V, solve for W (linear)
                
                # For fixed V and W:
                # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = delta[i,j,k]
                # Let B[i*m+j, c] = sum over... no, this doesn't separate cleanly.
                
                # Actually, for fixed V and W, define for each (i,k):
                # M[i*m+j, c] needs to satisfy: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta[i,j,k]
                # This is: for each (i,j,k), a linear equation in U[i*m+j, :]
                # But U[i*m+j, :] appears in equations for all k.
                
                # Let me restructure. For fixed V, W:
                # The (i,j,k) equation is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta[i,j,k]
                # Group by (i,j): for each (i,j), we have p equations (one per k) in r unknowns U[i*m+j, 0..r-1]
                # So for each (i,j), we solve a p x r system: A[i,j] @ U[i*m+j, :] = b[i,j]
                # where A[i,j][k,c] = V[j*p+k,c] * W[k*n+i,c]
                # and b[i,j][k] = delta[i,j,k]
                
                # This works! Let's implement alternating least squares properly.
                
                # Solve for U given V, W
                for i in range(n):
                    for j in range(m):
                        # Build A (p x r) and b (p)
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.zeros(p)
                        # delta[i,j,k] = 1 if i==j==k... no, delta is the identity in the matrix mult sense
                        # Actually the tensor T[i,j,k] = 1 if this is a "valid" multiplication term
                        # For matrix multiplication (A x B), C[i,k] = sum_j A[i,j] * B[j,k]
                        # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor)
                        # Wait, no. The tensor for matmul is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
                        # Because C[i,k] = sum_j A[i,j] B[j,k], so every (i,j,k) triple contributes.
                        b[:] = 1.0
                        # Solve A @ u = b in least squares
                        u, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_t[i * m + j, :] = u
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        v, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_t[j * p + k, :] = v
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        w, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_t[k * n + i, :] = w
                
                # Check convergence: compute reconstruction error
                # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Should equal 1 for all i,j,k
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_t[i * m + j, :], V_t[j * p + k, :] * W_t[k * n + i, :])
                            error += (val - 1.0) ** 2
                
                if error < 1e-12:
                    converged = True
                    break
                
                # Check for NaN/Inf
                if not (np.all(np.isfinite(U_t)) and np.all(np.isfinite(V_t)) and np.all(np.isfinite(W_t))):
                    break
            
            if converged:
                # Verify exactly
                verified = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_t[i * m + j, :], V_t[j * p + k, :] * W_t[k * n + i, :])
                            if abs(val - 1.0) > 1e-10:
                                verified = False
                                break
                        if not verified:
                            break
                    if not verified:
                        break
                
                if verified:
                    return U_t, V_t, W_t
        
        return None
    
    # Try ranks from lower bound upward
    # Start with a few promising ranks
    ranks_to_try = []
    # Add some candidate ranks
    for r in range(lower_bound, min(rank_school, lower_bound + 30)):
        ranks_to_try.append(r)
    
    # Also try some specific ranks that might work
    for r in [30, 32, 34, 36, 38, 40]:
        if lower_bound <= r < rank_school and r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try.sort()
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
            
        result = try_rank(r, num_trials=3, max_iter=100)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_rank = r
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            # If we found a solution, try to go lower
            break  # Restart the loop with lower ranks
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest 0.5 if close
    def to_list(mat):
        mat_list = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_list.append(rounded)
                else:
                    row_list.append(float(val))
            mat_list.append(row_list)
        return mat_list
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
