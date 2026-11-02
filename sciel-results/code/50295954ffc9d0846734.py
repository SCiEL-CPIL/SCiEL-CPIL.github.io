import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build the target tensor T of shape (n, m, p) where T[i,j,k] = 1
    # The decomposition requires: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k]
    # Wait, the convention is W[k*n+i, r], so W is indexed by (k, i)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor equation: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to find a lower rank decomposition
    # Use random initialization and iterative refinement
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # First, create the schoolbook solution as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Use a greedy/iterative approach
    
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that the tensor product equals the target
    
    # Let's try ranks from lower to schoolbook
    # Use a simple iterative method: initialize randomly, then use ALS (alternating least squares)
    
    def verify_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half_int(arr, tol=1e-4):
        """Round array to nearest half-integer and verify"""
        rounded = np.round(arr * 2) / 2
        return rounded
    
    # Try various ranks
    for rank in range(1, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 5 if rank > 50 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating minimization
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system for each (i,j)
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        # Let w_k,c = V[j*p+k,c] * W[k*n+i,c]
                        # Then sum_c U[i*m+j,c] * w_k,c = 1 for k=0..p-1
                        # This is: W_mat @ U[i*m+j,:] = 1 where W_mat[k,c] = V[j*p+k,c]*W[k*n+i,c]
                        W_mat = np.zeros((p, rank))
                        for k in range(p):
                            W_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve W_mat @ u = 1
                        try:
                            u_sol = np.linalg.lstsq(W_mat, np.ones(p), rcond=None)[0]
                            U[i*m+j, :] = u_sol
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        V_mat = np.zeros((n, rank))
                        for i in range(n):
                            V_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v_sol = np.linalg.lstsq(V_mat, np.ones(n), rcond=None)[0]
                            V[j*p+k, :] = v_sol
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        W_mat2 = np.zeros((m, rank))
                        for j in range(m):
                            W_mat2[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w_sol = np.linalg.lstsq(W_mat2, np.ones(m), rcond=None)[0]
                            W[k*n+i, :] = w_sol
                        except:
                            pass
                
                # Check convergence
                if verify_decomp(U, V, W, tol=1e-4):
                    # Found a valid decomposition!
                    # Try to round to half-integers
                    U_round = round_to_half_int(U)
                    V_round = round_to_half_int(V)
                    W_round = round_to_half_int(W)
                    if verify_decomp(U_round, V_round, W_round, tol=1e-6):
                        best_rank = rank
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        break
                    else:
                        # Use the non-rounded version
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        break
                elif verify_decomp(U, V, W, tol=1e-3):
                    # Close, try to polish
                    U_round = round_to_half_int(U)
                    V_round = round_to_half_int(V)
                    W_round = round_to_half_int(W)
                    if verify_decomp(U_round, V_round, W_round, tol=1e-6):
                        best_rank = rank
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        break
            
            if best_rank == rank:
                break
        
        if best_rank == rank:
            break
    
    # Convert to lists
    if best_U is not None and best_rank < schoolbook_rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
