"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin
    
    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def make_lists(U_arr, V_arr, W_arr):
        """Convert numpy arrays to list format with clean values."""
        def clean_matrix(arr):
            result = []
            for row in arr:
                clean_row = []
                for val in row:
                    if abs(val) < 1e-12:
                        clean_row.append(0)
                    else:
                        # Round to nearest half-integer or integer if close
                        v = round(val * 2) / 2
                        if abs(val - v) < 1e-10:
                            clean_row.append(v)
                        else:
                            clean_row.append(float(val))
                result.append(clean_row)
            return result
        return {"U": clean_matrix(U_arr), "V": clean_matrix(V_arr), "W": clean_matrix(W_arr)}
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = make_lists(U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try to find lower rank decompositions using optimization
    # Strategy: for a given rank R, we need to find U, V, W such that
    # for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # 
    # This is a system of trilinear equations. We can fix one matrix and solve
    # for the others iteratively.
    
    def try_optimize(R, seed, max_iter=200):
        """Try to find a rank-R decomposition using alternating minimization."""
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # Constraint: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            # Let A[i*m+j, r] = V[j*p+k,r] * W[k*n+i,r] for each k
            # We need U[i*m+j, :] @ A_k = 1 for all k
            # Stack: [A_0; A_1; ...; A_{p-1}] @ U[i*m+j, :]^T = [1, 1, ..., 1]^T
            
            for i in range(n):
                for j in range(m):
                    # Build linear system
                    # For each k, we have: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b_vec = np.ones(p)
                    # Solve A_mat @ U_row^T = b_vec
                    try:
                        U_row, residuals, rank_r, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[i*m+j, :] = U_row
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b_vec = np.ones(n)
                    try:
                        V_row, residuals, rank_r, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[j*p+k, :] = V_row
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b_vec = np.ones(m)
                    try:
                        W_row, residuals, rank_r, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[k*n+i, :] = W_row
                    except:
                        pass
        
        # Check if we converged
        if verify(U, V, W, tol=1e-8):
            return make_lists(U, V, W)
        return None
    
    # Try ranks from 60 down
    for R in range(rank_sb - 1, 35, -1):
        if time.time() > deadline - 5:
            break
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = try_optimize(R, seed, max_iter=100)
            if result is not None:
                best_result = result
                best_rank = R
                break
        if best_rank < rank_sb:
            # Found improvement, try to go even lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END

