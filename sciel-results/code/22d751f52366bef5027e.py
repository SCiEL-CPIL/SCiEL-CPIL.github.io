import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T
    # T[a, b, c] where a is index in U, b in V, c in W
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if b's first index == a's second index AND c's first index == b's second index AND c's second index == a's first index
    # Actually, the tensor is 1 only for the specific triples corresponding to the multiplication terms.
    # In the seed, for each i,j,k, U[i*m+j], V[j*p+k], W[k*n+i] are set to 1.
    # This implies T[(i*m+j), (j*p+k), (k*n+i)] = 1.
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # We want to find U, V, W of rank R such that sum_r U[:,r] V[:,r] W[:,r] = T
    # Flatten T to a vector of length dim_u * dim_v * dim_w?
    # The equation is: for all a,b,c: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # Strategy: Fix U, V. Solve for W.
    # For each c, we have a system:
    # sum_r W[c,r] (U[a,r] V[b,r]) = T[a,b,c] for all a,b.
    # Let X be a matrix of size (dim_u * dim_v) x R.
    # X[a*dim_v + b, r] = U[a,r] V[b,r]
    # Then X w_c = t_c, where t_c is a vector of length dim_u * dim_v.
    # t_c[a*dim_v + b] = T[a,b,c]
    
    def try_rank(R, max_iters=100, seed=0):
        rng = np.random.RandomState(seed)
        for it in range(max_iters):
            U = rng.randn(dim_u, R)
            V = rng.randn(dim_v, R)
            
            # Precompute X matrices? No, X depends on c? No, X is same for all c?
            # X[a*dim_v + b, r] = U[a,r] V[b,r]
            # This matrix X is independent of c!
            # So we compute X once.
            # X is (dim_u * dim_v) x R
            
            # Construct X
            # U is (dim_u, R), V is (dim_v, R)
            # We need outer product of U and V?
            # X[i, j, r] = U[i, r] V[j, r]
            # Flatten i, j to row index.
            
            # Efficient construction:
            # X = np.einsum('ir,jr->ijr', U, V).reshape(dim_u * dim_v, R)
            
            X = np.einsum('ir,jr->ijr', U, V).reshape(dim_u * dim_v, R)
            
            # Now for each c, solve X w_c = t_c
            # t_c is T[:, :, c] flattened
            
            W = np.zeros((dim_w, R))
            success = True
            for c in range(dim_w):
                t_c = T[:, :, c].flatten()
                # Solve X w_c = t_c
                # Since R < dim_u * dim_v, it's overdetermined.
                # Use least squares.
                try:
                    w_c, residuals, rank_x, s = np.linalg.lstsq(X, t_c, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                # Check if solution is exact
                # Residual should be close to 0
                if residuals.size > 0 and residuals[0] > 1e-9:
                    success = False
                    break
                
                W[c, :] = w_c
            
            if success:
                # Verify full reconstruction
                # Reconstruct T_recon = sum_r U[:,r] V[:,r] W[:,r]
                # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Check if T_recon == T
                # We can check a few random entries or all.
                # All is 216*216*216 = ~10M ops, fast enough.
                
                # Efficient check:
                # Compute T_recon and compare with T.
                # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                # This might be memory heavy (216^3 floats = 10M * 8 bytes = 80MB). OK.
                
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-9):
                    return U, V, W
            
        return None

    # Try ranks from low to high
    # Known bound for 6x6 is ~123. Let's start from 120.
    # If 120 fails, try 125, 130, etc.
    # To save time, we can start from a higher rank if we are not confident.
    # But we want the minimum.
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    best_result = None
    best_rank = None
    
    # Try a few ranks
    # If we find a solution at R, we can try to reduce it?
    # Or just try R = 120, 121, ...
    
    # Let's try R = 123 first (known bound)
    # If it fails, try 125, 130...
    
    for R in range(120, 160):
        if time.time() > deadline:
            break
            
        res = try_rank(R, max_iters=50, seed=R)
        if res is not None:
            U, V, W = res
            best_rank = R
            best_result = (U, V, W)
            break # Found the lowest rank in this sequence
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
