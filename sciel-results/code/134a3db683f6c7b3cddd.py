"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=i*p+k for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W index: k*n+i per the convention
                T[a, b, c] = 1.0
    
    # Try to find a rank-R decomposition
    # We'll try decreasing R from some upper bound
    best_result = None
    best_rank = n * m * p
    
    # Start with schoolbook
    U = np.zeros((n*m, best_rank))
    V = np.zeros((m*p, best_rank))
    W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    def check(U, V, W, tol=1e-10):
        R = U.shape[1]
        # Compute the reconstructed tensor
        # For efficiency, do it in chunks
        err = 0.0
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # outer product u ⊗ v ⊗ w
            # This is (n*m) x (m*p) x (p*n)
            # We can compute as: for each a,b, sum_r u[a]*v[b]*w[c]
            # But let's just check the non-zero entries
        # Check all entries
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    if check(U, V, W):
        best_result = (U.copy(), V.copy(), W.copy())
        best_rank = best_rank
    
    # Try to reduce rank using optimization
    # For each target rank, try to find a solution
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
        
        # Initialize with random values
        np.random.seed(42 + target_rank)
        U_try = np.random.randn(n*m, target_rank)
        V_try = np.random.randn(m*p, target_rank)
        W_try = np.random.randn(p*n, target_rank)
        
        # Use a simple gradient descent or least squares
        # The objective is: minimize sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
        # This is a non-convex problem.
        
        # For small target_rank, try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            np.random.seed(1000 + target_rank + restart)
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Scale to make the product ~1
            U_r /= np.linalg.norm(U_r, axis=0, keepdims=True) + 1e-10
            V_r /= np.linalg.norm(V_r, axis=0, keepdims=True) + 1e-10
            W_r /= np.linalg.norm(W_r, axis=0, keepdims=True) + 1e-10
            
            # Check if this random init happens to work (unlikely)
            if check(U_r, V_r, W_r):
                best_result = (U_r.copy(), V_r.copy(), W_r.copy())
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    if best_result is None:
        # Return schoolbook
        U, V, W = U, V, W
    else:
        U, V, W = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
