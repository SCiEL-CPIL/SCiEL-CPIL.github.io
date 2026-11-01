import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Target Tensor T for trace(ABC)
    # T[i, j, k, i', j', k'] is 1 if j==j', k==k', i'==i
    # In the flattened index scheme used by U, V, W:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The tensor entry T[U_idx, V_idx, W_idx] is 1 if:
    # (i*m+j) corresponds to i,j
    # (j*p+k) corresponds to j',k
    # (k*n+i) corresponds to k',i'
    # And j==j', k==k', i'==i.
    # Note: The W index is k*n + i. So W_idx = k*n + i.
    # The condition is:
    # j (from U) == j' (from V)
    # k (from V) == k' (from W)
    # i' (from W) == i (from U)
    
    # Let's precompute the target tensor as a sparse representation or a dense array.
    # Size: (n*m) x (m*p) x (p*n)
    # 9 x 12 x 12 = 1296 elements. Small enough for dense.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def compute_error(U, V, W):
        """
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        Computes || sum_c U_c o V_c o W_c - T ||^2
        """
        R = U.shape[1]
        # S[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor contraction.
        # We can do this by summing outer products.
        # For small R and small dimensions, explicit loop or einsum is fine.
        # einsum: 'ar, br, cr -> abc'
        S = np.einsum('ar,br,cr->abc', U, V, W)
        diff = S - T
        return np.sum(diff**2)

    def objective(params):
        R = params[0] # This won't work, R is fixed per call
        # We will define objective inside the loop where R is known
        pass

    def verify_solution(U, V, W):
        """
        Verifies if the rounded solution exactly matches T.
        """
        R = U.shape[1]
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=1e-9)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_solution = schoolbook()
    best_rank = n * m * p

    # Try to find lower ranks
    # Theoretical lower bound for trace tensor is roughly n*m*p / something?
    # For 3x3x3, rank is 23.
    # For 3x3x4, let's try ranks from 20 to 35.
    
    for target_rank in range(20, 36):
        if time.time() > deadline:
            break
        
        num_vars = (n*m + m*p + p*n) * target_rank
        
        def obj(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            err = compute_error(U, V, W)
            # Add sparsity penalty to encourage simple solutions
            penalty = 0.1 * (np.sum(np.abs(U)) + np.sum(np.abs(V)) + np.sum(np.abs(W)))
            return err + penalty

        # Multiple restarts
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Scale: values in T are 1. U,V,W should be O(1).
            x0 = np.random.randn(num_vars) * 0.5
            
            try:
                res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    U_flat = res.x[:n*m*target_rank]
                    V_flat = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                    W_flat = res.x[n*m*target_rank + m*p*target_rank:]
                    
                    U = U_flat.reshape((n*m, target_rank))
                    V = V_flat.reshape((m*p, target_rank))
                    W = W_flat.reshape((p*n, target_rank))
                    
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    if verify_solution(U_rounded, V_rounded, W_rounded):
                        # Found a valid solution!
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
            except Exception:
                pass

    return best_solution

# EVOLVE-BLOCK-END
