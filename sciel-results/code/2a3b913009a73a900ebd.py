import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_trivial = n * m * p
    rank_target = 17
    
    # Helper to flatten tensor
    def tensor_to_flat(U, V, W):
        # U: n*m x rank, V: m*p x rank, W: p*n x rank
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Note: W index is k*n + i based on task description "W[k*n+i,r]"
        # Wait, task says: "output factor follows the official trace convention W[k*n+i,r]"
        # And shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # U index: i*m + j? Or i*p + j? 
        # Standard: A is n x m. B is m x p. C is n x p.
        # U corresponds to A: size n*m. Index i*m + j.
        # V corresponds to B: size m*p. Index j*p + k.
        # W corresponds to C: size n*p. Index i*p + k? 
        # Task says: "W[k*n+i,r]". This implies C is stored as C[k, i]? 
        # Or C is n x p, and index is k*n + i where k is column, i is row?
        # Let's check the shape: (p*n, rank).
        # If C is n x p, total elements n*p.
        # Index k*n + i implies k goes 0..p-1, i goes 0..n-1.
        # This is column-major order for C (if C is n x p, columns are length n).
        # So C[i, k] is stored at index k*n + i.
        
        # Let's construct the target tensor T_flat of size (n*m) * (m*p) * (p*n)?
        # No, the tensor is 3-mode.
        # Mode 1: A (n*m)
        # Mode 2: B (m*p)
        # Mode 3: C (p*n)
        
        # We want to find U, V, W such that:
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # where T is the matrix multiplication tensor.
        
        # T[a, b, c] is 1 if the product of A[a] and B[b] contributes to C[c].
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i  (based on W[k*n+i])
        
        # So T[a, b, c] = 1 if (a, b, c) satisfy:
        # i = a // m, j = a % m
        # j' = b // p, k = b % p  -- Wait.
        # B is m x p. Index b = j*p + k. So j = b // p, k = b % p.
        # For the product to be non-zero, the inner indices must match: j == j'.
        # And the output index c must be k*n + i.
        
        # So T[a, b, c] = 1 if:
        # 1. a = i*m + j
        # 2. b = j*p + k  (Note: j from a must equal j from b)
        # 3. c = k*n + i
        
        # Let's build the target tensor T (sparse or dense).
        # Size: (n*m) * (m*p) * (p*n)
        # For 2,3,3: 6 * 9 * 6 = 324 elements. Small enough for dense.
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Reshape T to a matrix for verification?
        # We can verify by checking:
        # sum_r U[:, r] V[:, r] W[:, r] == T (broadcasting)
        
        def error(params):
            # params: flattened U, V, W
            # U: n*m x rank_target
            # V: m*p x rank_target
            # W: p*n x rank_target
            size_u = n * m * rank_target
            size_v = m * p * rank_target
            size_w = p * n * rank_target
            
            U_flat = params[:size_u]
            V_flat = params[size_u:size_u+size_v]
            W_flat = params[size_u+size_v:]
            
            U = U_flat.reshape(n*m, rank_target)
            V = V_flat.reshape(m*p, rank_target)
            W = W_flat.reshape(p*n, rank_target)
            
            # Compute T_est
            # T_est[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # This is a tensor contraction.
            # Efficient way:
            # T_est = np.einsum('ar,br,cr->abc', U, V, W)
            
            T_est = np.einsum('ar,br,cr->abc', U, V, W)
            
            return np.sum((T_est - T)**2)

        # Try to find a solution
        # Random restarts
        best_rank = rank_trivial
        best_solution = None
        
        # Generate Rank 18 solution (always valid)
        U18 = np.zeros((n*m, rank_trivial))
        V18 = np.zeros((m*p, rank_trivial))
        W18 = np.zeros((p*n, rank_trivial))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U18[a, col] = 1
                    V18[b, col] = 1
                    W18[c, col] = 1
                    col += 1
        
        # Check Rank 17
        # We will try a few random initializations
        for restart in range(20):
            np.random.seed(restart)
            # Initialize with small random integers? Or reals?
            # Reals are easier to optimize, then round.
            U_init = np.random.randn(n*m, rank_target)
            V_init = np.random.randn(m*p, rank_target)
            W_init = np.random.randn(p*n, rank_target)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use Nelder-Mead for local optimization
            try:
                res = minimize(error, params_init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.success or res.fun < 1e-6:
                    # Check if we can round to integers/half-integers
                    U_opt = res.x[:n*m*rank_target].reshape(n*m, rank_target)
                    V_opt = res.x[n*m*rank_target : n*m*rank_target + m*p*rank_target].reshape(m*p, rank_target)
                    W_opt = res.x[n*m*rank_target + m*p*rank_target :].reshape(p*n, rank_target)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify exactness with rounded values
                    T_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_rounded, T, atol=1e-9):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            except Exception:
                pass
        
        # Fallback to Rank 18
        return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
# EVOLVE-BLOCK-END
