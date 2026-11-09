"""Matrix multiplication 2x4x6 rank-minimal decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    DEADLINE = time.time() + 125  # safety margin before 150s kill

    # Schoolbook (rank 48) as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    def to_list(mat):
        # Convert numpy array to list of lists with Fraction entries
        return [[Fraction(x) for x in row] for row in mat]

    def verify(U, V, W, tol=1e-7):
        """Check if U,V,W with given rank exactly decompose the matmul tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        r = U.shape[1]
        # Build the tensor check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Reshape for batch computation
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m + j, :]
                    v_val = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    s = np.dot(np.multiply(u_val, v_val), w_val)
                    if abs(s - 1.0) > 1e-6:
                        errors += 1
        return errors == 0

    def try_decomposition(U, V, W, max_iter=300):
        """Iteratively refine U,V,W to satisfy the tensor equation."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                return None
            
            # Update U: for each (i,j), U[i*m+j,:] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j,:]
            # Let A[i*m+j] = V[j*p+k,:] * W[k*n+i,:] (element-wise) for each k
            # We need A @ U[i*m+j,:] = ones(p)
            # Stack: p equations, r unknowns
            
            for i in range(n):
                for j in range(m):
                    # Build p x r matrix
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if verify_fast(U, V, W):
                return (U, V, W)
        
        return None

    # Try various ranks from low to high
    best_result = None
    np.random.seed(42)
    
    for rank in range(16, 48):
        if time.time() > DEADLINE - 10:
            break
        
        # Try multiple random initializations
        n_trials = 3
        for trial in range(n_trials):
            if time.time() > DEADLINE - 5:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            result = try_decomposition(U, V, W, max_iter=200)
            if result is not None:
                U_r, V_r, W_r = result
                # Verify once more
                if verify_fast(U_r, V_r, W_r):
                    best_result = (U_r, V_r, W_r, rank)
                    break
        if best_result is not None:
            break
    
    if best_result is not None:
        U_r, V_r, W_r, rank = best_result
        return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    
    # Fallback: schoolbook
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
