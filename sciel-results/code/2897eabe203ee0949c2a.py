import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Construct the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def compute_error(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Error = || sum_c U[:,c] V[:,c]^T W[:,c]^T - T ||_F
        # We can compute the reconstructed tensor R
        # R[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        # This is a tensor contraction.
        # Using einsum: 'ak,bk,ck->abc'
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.linalg.norm(R - T)

    def optimize(r, init_U=None, init_V=None, init_W=None, max_iters=5000, lr=0.01):
        if init_U is None:
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
        else:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
            
        # Simple gradient descent
        # Gradient of error^2 w.r.t U, V, W
        # E = ||R - T||_F^2
        # dE/dU[a,k] = 2 * sum_{b,c} (R[a,b,c] - T[a,b,c]) V[b,k] W[c,k]
        
        for it in range(max_iters):
            R = np.einsum('ak,bk,ck->abc', U, V, W)
            diff = R - T
            err = np.linalg.norm(diff)
            if err < 1e-6:
                return U, V, W, err
            
            # Compute gradients
            # dU[a,k] = 2 * sum_{b,c} diff[a,b,c] V[b,k] W[c,k]
            # This can be done efficiently
            # diff is (n*m, m*p, p*n)
            # V is (m*p, r), W is (p*n, r)
            # We want (n*m, r)
            # grad_U = 2 * np.einsum('abc,bk,ck->ak', diff, V, W)
            grad_U = 2 * np.einsum('abc,bk,ck->ak', diff, V, W)
            grad_V = 2 * np.einsum('abc,ak,ck->bk', diff, U, W)
            grad_W = 2 * np.einsum('abc,ak,bk->ck', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: normalization to prevent divergence
            # U = U / np.linalg.norm(U) * np.linalg.norm(U) # No, just keep scale
            
        return U, V, W, err

    # Start with schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to reduce rank
    start_time = time.time()
    for r in range(rank_schoolbook - 1, 35, -1):
        if time.time() - start_time > 140:
            break
            
        found = False
        for restart in range(5):
            if restart == 0:
                # Initialize with schoolbook terms, drop one
                U_init = U_school[:, :r]
                V_init = V_school[:, :r]
                W_init = W_school[:, :r]
            else:
                U_init = None
                V_init = None
                W_init = None
                
            U, V, W, err = optimize(r, U_init, V_init, W_init, max_iters=1000, lr=0.05)
            
            if err < 1e-6:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
        
        if found:
            # Try to reduce further immediately? The loop does it.
            pass

    # Convert to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
