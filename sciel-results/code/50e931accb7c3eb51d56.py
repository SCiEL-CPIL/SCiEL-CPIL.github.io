"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i, j, k] = delta for A[i,j]*B[j,k]
    # The decomposition gives: for each c, U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Sum over c should equal 1 if (i,j,k) is a valid multiplication term, else 0
    # Actually, the tensor T_{ijk} = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the standard formulation, T[i,j,k] = 1 always (for the multiplication tensor)
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    
    # Let's build T explicitly
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # Each multiplication term A[i,j]*B[j,k] contributes to C[i,k]
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == T[i,j,k]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - T[i,j,k])
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use schoolbook for the base, then try to find a lower-rank decomposition
    # using optimization. The schoolbook rank is n*m*p = 2*4*5 = 40.
    # 
    # For 2x4x5, we can think of it as block structure.
    # A is 2x4, B is 4x5, C is 2x5.
    # 
    # Let's try a structured approach:
    # Split A into two 2x2 blocks: A = [A1 A2] where A1, A2 are 2x2
    # Split B into two 4x2.5... that doesn't work cleanly.
    # 
    # Alternative: use the fact that rank of 2x2x2 is 7 (Strassen), 
    # and see if we can build up from there.
    # 
    # For now, let's use schoolbook as baseline and try local optimization to reduce rank.
    
    # Start with schoolbook
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    err0 = _verify(U0, V0, W0, n, m, p)
    assert err0 < 1e-10, f"Schoolbook verification failed: {err0}"
    
    # Try to reduce rank by setting some columns to zero and optimizing the rest
    # Use a greedy approach: try removing one column at a time and see if the residual
    # can be absorbed by the remaining columns.
    
    # Actually, let's try a different approach: start with a lower rank and use
    # gradient-based optimization to find a valid decomposition.
    
    # The multiplication tensor for 2x4x5 has rank at least max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # and at most n*m*p = 40.
    
    # Let's try ranks from 20 upward and see if we can find a valid decomposition.
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    
    for target_rank in range(20, rank0):
        if time.time() > deadline - 20:
            break
        
        # Initialize with random or structured matrices
        np.random.seed(42)
        
        # Try multiple initializations
        for init in range(5):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Simple iterative optimization: minimize sum of squared errors
            # T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual for each (i,j,k)
                # Update U, V, W using gradient descent or ALS
                
                # ALS: fix V and W, update U
                for c in range(target_rank):
                    # For each (i,j), we want U[i*m+j, c] such that
                    # sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] approximates T[i,j,k]
                    # With U fixed except for column c, this is linear in U[:,c]
                    # But it's cubic overall, so let's do a simpler approach.
                    pass
                
                # Use a simple gradient step
                # dL/dU[i*m+j, c] = sum_k (residual[i,j,k]) * V[j*p+k, c] * W[k*n+i, c]
                # But residual depends on all U, V, W, so this is coupled.
                
                # Let's use a different strategy: use the known structure.
                # For the schoolbook, each column corresponds to one (i,j,k) term.
                # To reduce rank, we need to combine terms.
                
                break  # Skip to next initialization
            
            # Check if this random initialization happens to be valid (unlikely)
            err = _verify(U, V, W, n, m, p)
            if err < 1e-8:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break
        
        if best_rank < rank0:
            break
    
    # If we found a better rank, use it; otherwise fall back to schoolbook
    if best_rank < rank0:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U0
        V_out = V0
        W_out = W0
    
    # Convert to lists with integer/half-integer entries if possible
    # The schoolbook has all 1s, which is fine.
    
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
