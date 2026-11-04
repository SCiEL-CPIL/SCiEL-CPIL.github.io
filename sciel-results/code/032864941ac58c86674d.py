"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build the tensor T[i, j, k] = 1 for all valid (i,j,k)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is all ones for 0<=i<n, 0<=j<m, 0<=k<p
    
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Strategy: try various target ranks and use optimization
    
    def verify_half_integer(U, V, W, tol=1e-6):
        """Check if decomposition is exact with half-integer entries."""
        # Check entries are half-integers
        for arr in [U, V, W]:
            for val in arr:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
        # Verify decomposition
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_integer(arr):
        return np.round(arr * 2) / 2.0
    
    # Try target ranks from best_rank-1 down to a reasonable minimum
    # Use random initialization and Newton-like refinement
    
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, 27, -1):  # 27 is below lower bound
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        n_attempts = 5 if target_rank > 80 else 10
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Iterative refinement: minimize error
            # Use a simple gradient-free approach: coordinate descent
            # or use the fact that for fixed U,V, optimal W can be computed
            
            # Actually, let's use a different approach: 
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of equations.
            
            # Let's try a structured approach: use known small decompositions
            # For 4x4x4, try rank 49 (known achievable)
            # For the 4x4x3 part, try various ranks
            
            if target_rank == 49 + 48:  # 97: 49 for 4x4x4 + 48 for 4x4x3 schoolbook
                # This is just combining known results, not really an improvement
                pass
        
        # After trying random, try structured decomposition
        # Split: first 4 columns of output (4x4) and last 3 columns (4x3)
        # For 4x4x4: use a known low-rank decomposition
        # For 4x4x3: use schoolbook or try to improve
        
        if target_rank >= 97:
            # Try: rank 49 for 4x4x4 + rank 48 for 4x4x3 = 97
            # But we need to construct actual matrices
            
            # 4x4x4 decomposition: Let's use a simple structured one
            # Actually, let's just try to optimize numerically
            
            # Use numpy lstsq approach
            # Build the linear system for W given U,V
            # Or iterate: fix two, solve for third
            
            # Let's do coordinate descent
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            # Precompute for speed
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                # W[k*n+i, :] = argmin sum_{j} (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
                # This is linear in W[k*n+i, :]
                # Let M[k,i] be a matrix where M[k,i,j,c] = U[i*m+j,c]*V[j*p+k,c]
                # Then we need M[k,i] @ W[k*n+i,:] = ones(m)
                
                # Build the system
                # For fixed (k,i), the equation is:
                # sum_c W[k*n+i,c] * (sum_j delta_{target,j} * U[i*m+j,c]*V[j*p+k,c]) = 1 for each j
                # Wait, no. For each (i,j,k):
                # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                
                # Group by (k,i): for fixed (k,i), we have m equations (one per j)
                # and target_rank unknowns (W[k*n+i, c] for c in range(target_rank))
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # Then A @ W[k*n+i, :] = ones(m)
                
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                        # Solve A_mat @ w = ones(m)
                        try:
                            w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W_curr[k*n+i, :] = w_sol
                        except:
                            pass
                
                # Update U: for each (i,j), solve
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                        try:
                            u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U_curr[i*m+j, :] = u_sol
                        except:
                            pass
                
                # Update V: for each (j,k), solve
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                        try:
                            v_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V_curr[j*p+k, :] = v_sol
                        except:
                            pass
                
                # Check convergence
                if iteration % 10 == 9:
                    # Compute error
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
                                max_err = max(max_err, abs(s - 1.0))
                    if max_err < 1e-4:
                        # Try rounding
                        U_rounded = round_to_half_integer(U_curr)
                        V_rounded = round_to_half_integer(V_curr)
                        W_rounded = round_to_half_integer(W_curr)
                        if verify_half_integer(U_rounded, V_rounded, W_rounded, tol=1e-3):
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                            best_rank = target_rank
                        break
            else:
                continue
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
