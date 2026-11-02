import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Target tensor T[i, j, k, i', k'] = 1 if i==i' and k==k'
    # Indices:
    # i, i' in [0, n)
    # j in [0, m)
    # k, k' in [0, p)
    # U index: i*m + j
    # V index: j*p + k'  (Note: V is (m*p, rank), so index is j*p + k')
    # W index: k*n + i'  (Note: W is (p*n, rank), so index is k*n + i')
    # Wait, the problem statement says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Output factor W[k*n+i, r] follows official trace convention.
    # Standard tensor decomposition for matmul C = A B:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] = 1 if ...?
    # The tensor for matrix multiplication $M_{n \times m, m \times p}$ is usually defined as:
    # T[i, j, k, i', k'] is not standard.
    # Let's stick to the prompt's implied structure:
    # sum_alpha U[i'm+j, alpha] * V[jp+k', alpha] * W[kn+i, alpha] == T[i, j, k, i', k']
    # where T[i, j, k, i', k'] = 1 if i==i' and k==k', else 0.
    # This matches C[i, k] = sum_j A[i, j] B[j, k].
    # A is n x m, B is m x p, C is n x p.
    # i is row of A/C (0..n-1)
    # j is col of A / row of B (0..m-1)
    # k is col of B/C (0..p-1)
    # i' is row of A/C? No, i' is the row index in the output?
    # Let's re-read: T[i, j, k, i', k']
    # If i==i' and k==k', it is 1.
    # So the tensor is non-zero when the input row index matches the output row index AND the input col index matches the output col index.
    # This is the standard matmul tensor.
    
    # Dimensions of T:
    # i: 0..n-1
    # j: 0..m-1
    # k: 0..p-1
    # i': 0..n-1
    # k': 0..p-1
    # Total size: n * m * p * n * p = 3 * 4 * 5 * 3 * 5 = 900 entries.
    
    # We want to find U (n*m, r), V (m*p, r), W (p*n, r)
    # such that for all i, j, k, i', k':
    # sum_alpha U[i*m+j, alpha] * V[j*p+k', alpha] * W[k*n+i', alpha] == T[i, j, k, i', k']
    
    # Note the indices in the sum:
    # U index: i*m + j
    # V index: j*p + k'
    # W index: k*n + i'
    
    # Let's construct the target tensor T_flat of length n*m*p*n*p
    # But we can compute the error on the fly or flatten it.
    
    # For optimization, we flatten U, V, W into a single vector x.
    # Size of x: r*(n*m + m*p + p*n)
    
    # We will try ranks r from 55 down to 30 (arbitrary range below 60)
    # If we find an exact solution, return it.
    
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_sb
    
    # Precompute indices for T
    # T[i, j, k, i', k']
    # We can create a 5D array T
    T = np.zeros((n, m, p, n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i, j, k, i, k] = 1.0
                
    # Function to compute error given x (flattened U, V, W)
    def error_func(x, r):
        # Unpack
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = x[0:size_U].reshape(n, m, r)
        V = x[size_U:size_U+size_V].reshape(m, p, r)
        W = x[size_U+size_V:size_U+size_V+size_W].reshape(p, n, r)
        
        # Compute sum_alpha U[i, j, alpha] * V[j, k', alpha] * W[k, i', alpha]
        # Result R[i, j, k, i', k']
        # R[i, j, k, i', k'] = sum_alpha U[i, j, alpha] * V[j, k', alpha] * W[k, i', alpha]
        
        # Efficient computation using einsum
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        
        # We want R[i, j, k, i', k']
        # Let's contract U and V first?
        # U[i, j, a] * V[j, k', a] -> S[i, j, k', a]? No, sum over a.
        # S[i, j, k'] = sum_a U[i, j, a] * V[j, k', a]
        # This is (n, m, p)
        
        # Then R[i, j, k, i', k'] = sum_{k'}? No.
        # R[i, j, k, i', k'] = sum_a U[i, j, a] * V[j, k', a] * W[k, i', a]
        # This is a 5D tensor.
        
        # einsum: 'ija,jka,klb->ikjl'? No.
        # Indices:
        # U: i, j, a
        # V: j, k', a  (note: V second index is k')
        # W: k, i', a  (note: W first index is k, second is i')
        
        # We want output indices: i, j, k, i', k'
        # Sum over a.
        
        # Let's use np.einsum
        # U[i, j, a] * V[j, k', a] * W[k, i', a]
        # -> R[i, j, k, i', k']
        
        # 'ija,jka,kba->ijkba' ?
        # U: i, j, a
        # V: j, k, a  (using k for k')
        # W: k, b, a  (using b for i', k for k)
        # Wait, W index is k*n + i'. So W[k, i', a].
        # So W dimensions are (p, n, r). Index k is 0..p-1, i' is 0..n-1.
        
        # So:
        # U[i, j, a]
        # V[j, k, a]  (where k is k')
        # W[k, i', a] (where k is k, i' is i')
        
        # Sum over a.
        # Result R[i, j, k, i', k] ? No.
        # The output should be T[i, j, k, i', k'].
        # T depends on i, j, k, i', k'.
        
        # Let's map:
        # U[i, j, a]
        # V[j, k', a]
        # W[k, i', a]
        
        # Product P[i, j, k, i', k', a] = U[i, j, a] * V[j, k', a] * W[k, i', a]
        # Sum over a: R[i, j, k, i', k']
        
        # einsum: 'ija,jka,kla->ijkl'? No.
        # U: i, j, a
        # V: j, k, a  (k is k')
        # W: k, l, a  (k is k, l is i')
        # Result: i, j, k, l, k? No.
        # We need distinct indices for the output.
        # Output: i, j, k, i', k'
        
        # U[i, j, a]
        # V[j, k', a]
        # W[k, i', a]
        
        # Let's rename V's second index to kp (k_prime) and W's first index to k.
        # U[i, j, a]
        # V[j, kp, a]
        # W[k, i', a]
        
        # Result R[i, j, k, i', kp]
        
        # einsum: 'ija,jka,kla->ijkl' doesn't work because k is reused.
        # Use distinct letters.
        # U: i, j, a
        # V: j, m, a  (m is k')
        # W: k, l, a  (k is k, l is i')
        # Result: i, j, k, l, m
        
        # np.einsum('ija,jma,kl a->ijklm', U, V, W)
        # Note: spaces in einsum string are ignored? No, better to be careful.
        # 'ija,jma,kla->ijklm'
        
        R = np.einsum('ija,jma,kla->ijklm', U, V, W)
        
        # Compare with T
        # T[i, j, k, i', k']
        # R[i, j, k, i', k']
        
        diff = R - T
        return np.sum(diff * diff)
    
    # We will try to optimize for a few ranks
    # Start from rank 50 down to 40?
    # Each optimization takes time.
    
    for r in range(55, 30, -1):
        if time.time() > deadline:
            break
            
        # Initialize x
        # Use small random values
        np.random.seed(42) # Deterministic for reproducibility in search?
        # Better to use different seeds for different ranks
        
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = r * (size_U + size_V + size_W)
        
        x0 = np.random.randn(total_size) * 0.1
        
        # Optimize
        try:
            res = minimize(error_func, x0, args=(r,), method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-8})
            
            if res.success and res.fun < 1e-6:
                # Check for exactness with rounding
                x_opt = res.x
                U_flat = x_opt[0:size_U].reshape(size_U, r)
                V_flat = x_opt[size_U:size_U+size_V].reshape(size_V, r)
                W_flat = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(size_W, r)
                
                # Round to nearest 0.5
                U_round = np.round(U_flat * 2) / 2
                V_round = np.round(V_flat * 2) / 2
                W_round = np.round(W_flat * 2) / 2
                
                # Verify
                U_r = U_round.reshape(n, m, r)
                V_r = V_round.reshape(m, p, r)
                W_r = W_round.reshape(p, n, r)
                
                R_round = np.einsum('ija,jma,kla->ijklm', U_r, V_r, W_r)
                
                if np.allclose(R_round, T, atol=1e-9):
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    best_rank = r
                    break # Found a lower rank, stop
                    
        except Exception as e:
            pass
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
