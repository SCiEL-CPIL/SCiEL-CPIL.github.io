"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target: T[i,j,k] = 1 for all valid i,j,k
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Fallback: Schoolbook decomposition (Rank = n * m * p)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Efficient verification using tensor operations
    def verify(U, V, W, r):
        # For each (i,k), check if sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # We can compute this as: for each c, form outer product contributions
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                # For fixed (i,k), compute sum over j of the contribution
                total = 0.0
                for j in range(m):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    total += np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                err = abs(total - 1.0)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Try to find lower rank decompositions
    # Strategy: For fixed rank r, initialize U and V, solve for W via least squares
    # For each (k,i), we have m equations: sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
    
    # Key insight: We can try structured initializations that are more likely to work
    
    def try_rank(r, num_trials=20):
        """Try to find a rank-r decomposition."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Different initialization strategies
            if trial % 3 == 0:
                # Random Gaussian
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
            elif trial % 3 == 1:
                # Sparse-like initialization
                U = np.random.choice([0, 1], size=(nm, r)) * 0.5
                V = np.random.choice([0, 1], size=(mp, r)) * 0.5
            else:
                # Mixed initialization
                U = np.random.randn(nm, r) * 0.3
                V = np.random.randn(mp, r) * 0.3
                # Add some structure
                for c in range(r):
                    U[np.random.randint(0, nm), c] = 1.0
                    V[np.random.randint(0, mp), c] = 1.0
            
            # Solve for W
            W = np.zeros((pn, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A_ki of shape (m, r)
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    
                    # Solve A_ki @ w = ones(m)
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=1e-10)
                        W[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                continue
            
            # Verify
            err = verify(U, V, W, r)
            if err < 1e-8:
                return (U, V, W, r)
        
        return None
    
    # Search ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) is NOT a valid lower bound
    # But we know rank >= max(n, m, p) in some cases
    # Let's try from r=1 upward
    
    # First, quick check for rank 1 (separable case)
    # For rank 1: U[i*m+j] = a_i * b_j, V[j*p+k] = b_j * c_k, W[k*n+i] = c_k * a_i
    # Then U*V*W = a_i^2 * b_j^2 * c_k^2, need this = 1 for all i,j,k
    # This requires a_i, b_j, c_k to be constant, so set all to 1
    U1 = np.ones((nm, 1))
    V1 = np.ones((mp, 1))
    W1 = np.ones((pn, 1))
    err1 = verify(U1, V1, W1, 1)
    if err1 < 1e-10:
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    
    # Rank 1 doesn't work in general for non-separable tensors
    # Try ranks from 2 upward
    for r in range(2, min(60, n*m*p + 1)):
        if time.time() > deadline - 15:
            break
        
        # More trials for smaller ranks
        num_trials = 50 if r < 10 else 20 if r < 20 else 10
        
        result = try_rank(r, num_trials)
        if result is not None:
            best_result = result
            break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
