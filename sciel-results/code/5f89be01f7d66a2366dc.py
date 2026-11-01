import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 110

    # Trivial rank 44 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1

    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    # Target rank: 38 (Hypothesis: 2x2x11 has rank 38)
    # 2x2x2 rank is 7. 5 blocks of 2x2x2 = 35. 
    # Remaining dimension for k is 1. 
    # Standard tiling: 5*7 + 4 = 39.
    # Can we do better? 
    # If we can share some terms between the last "1" block and the previous blocks, 
    # or if the 2x2x11 tensor has lower rank than 5*7+4.
    # Actually, for 2x2xp, the rank is known to be 2p+2 for p>=2? No.
    # Rank(2,2,p) = 2p+2 is for p=2 (rank 6? No, 2x2x2 is 7).
    # Let's check known bounds.
    # R(2,2,2) = 7.
    # R(2,2,p) <= 2p + 5? 
    # If R(2,2,11) is 38, that would be a significant improvement over 39.
    
    target_rank = 38
    
    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unpack_params(x):
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U = x[:size_U].reshape(n * m, target_rank)
        V = x[size_U:size_U + size_V].reshape(m * p, target_rank)
        W = x[size_U + size_V:].reshape(p * n, target_rank)
        return U, V, W

    def residuals(x):
        U, V, W = unpack_params(x)
        # Tensor T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # a = i*m + j, b = j*p + k, c = k*n + i
        # Target: C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor for matrix mult is T_{i,j,k} = delta_{i,j} delta_{j,k}? No.
        # The identity is: (A @ B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # In tensor form: T_{i,j,k} is the structure tensor.
        # T_{i,j,k} = 1 if we are summing over j? 
        # The decomposition is: 
        # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = delta_{i,j} delta_{j,k} ? 
        # No, the standard definition of the matrix multiplication tensor T is:
        # T_{i, j, k} = 1 if i,j,k correspond to a valid multiplication term?
        # Actually, the tensor T is defined such that T_{a, b, c} = 1 if a,b,c match the indices of A,B,C in the product.
        # Specifically, for C = AB, the tensor is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
        # The identity is: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k} (with appropriate index contraction).
        # The decomposition requires:
        # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 if (i,j,k) is a "valid" triple?
        # Wait, the "valid" triple for the product C_{i,k} = sum_j A_{i,j} B_{j,k} involves the same j in A and B.
        # So the tensor T has T_{i,j,k} = 1 if the indices align?
        # Let's look at the indices in the problem statement:
        # U shape (n*m, rank) -> index a = i*m + j
        # V shape (m*p, rank) -> index b = j*p + k
        # W shape (p*n, rank) -> index c = k*n + i
        # The sum is over r.
        # The result is a tensor S_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}.
        # We require S_{a,b,c} to match the matrix multiplication tensor.
        # The matrix multiplication tensor M_{a,b,c} is 1 if:
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        # AND i=i', j=j', k=k'.
        # So M_{a,b,c} = 1 if i=a//m == c%n AND j=a%m == b//p AND k=b%p == c//n.
        # Otherwise 0.
        
        # We need to compute the error: S_{a,b,c} - M_{a,b,c}
        # Since we only need to match the specific structure, we can compute the residual for all a,b,c.
        # However, most entries of M are 0.
        # We can restrict the residual to the non-zero entries of M, or all of them.
        # There are n*m*p = 44 non-zero entries in M.
        # Total entries in S is (n*m)*(m*p)*(p*n) = 4 * 22 * 22 = 1936.
        # We only need to satisfy the 44 equations where M=1, and M=0 elsewhere?
        # Yes, S must equal M exactly.
        
        # Let's construct the list of (a,b,c) where M=1.
        # i in 0..n-1, j in 0..m-1, k in 0..p-1
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        
        # We will compute S for these specific (a,b,c) and subtract 1.
        # For all other (a,b,c), S must be 0.
        # To save time, we can just compute the full S tensor via einsum and subtract M.
        # S = np.einsum('ar,br,cr->abc', U, V, W)
        # M is a sparse tensor.
        
        # Efficient residual:
        # Only the 44 non-zero terms contribute to the "target" value 1.
        # The other 1892 terms must be 0.
        # So we need S_{a,b,c} = M_{a,b,c} for all a,b,c.
        
        # Let's just compute the full S and compare to M.
        S = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Construct M
        # M is zero everywhere except at (i*m+j, j*p+k, k*n+i) where it is 1.
        # We can initialize M as zeros and set the 1s.
        # But doing this inside the residual function is expensive.
        # Precompute the indices of the 1s in M.
        # We can't easily precompute M as a dense array of 1936 floats? Yes we can, it's small.
        
        # Pre-computation outside? 
        # Let's do it inside for clarity, but optimize.
        # Actually, let's just return the difference for the 44 non-zero spots and the 0s for the rest?
        # No, least_squares wants a vector of residuals.
        # We can flatten S and subtract flattened M.
        
        # To avoid creating M every time, we can note that M is fixed.
        # Let's create M once outside.
        
        return S.flatten() - M_flat

    # Precompute M_flat
    M = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
    M_flat = M.flatten()

    # Update residuals to use precomputed M_flat
    def residuals_opt(x):
        U, V, W = unpack_params(x)
        S = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return S.flatten() - M_flat

    # Try to find a solution for rank 38
    # Initialize with random values
    np.random.seed(42)
    
    found_38 = False
    
    # We can also try to initialize with the rank 39 solution and zero out one column?
    # Or just random search with multiple restarts.
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale: values around 1
        x0 = np.random.randn(n * m * target_rank + m * p * target_rank + p * n * target_rank)
        x0 *= 1.0
        
        try:
            res = least_squares(residuals_opt, x0, method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12)
            if res.success or res.cost < 1e-10:
                # Verify the solution
                U, V, W = unpack_params(res.x)
                
                # Check integer/half-integer constraint?
                # The problem allows integer/half-integer entries.
                # The solver might find floating point values close to integers/half-integers.
                # We should snap them to the nearest half-integer.
                
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness with snapped values
                S_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                if np.allclose(S_snap, M, atol=1e-9):
                    # Found a valid rank 38 decomposition!
                    best_rank = 38
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    found_38 = True
                    break
        except Exception:
            pass

    # If rank 38 not found, try rank 39 with a similar approach to ensure correctness
    # Or just use the trivial rank 44 if 39 wasn't found by the previous heuristic (which is gone now)
    # Let's add a quick check for rank 39 if 38 fails, using the same solver.
    
    if not found_38:
        target_rank_39 = 39
        def unpack_params_39(x):
            size_U = n * m * target_rank_39
            size_V = m * p * target_rank_39
            size_W = p * n * target_rank_39
            U = x[:size_U].reshape(n * m, target_rank_39)
            V = x[size_U:size_U + size_V].reshape(m * p, target_rank_39)
            W = x[size_U + size_V:].reshape(p * n, target_rank_39)
            return U, V, W

        def residuals_opt_39(x):
            U, V, W = unpack_params_39(x)
            S = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            return S.flatten() - M_flat

        for restart in range(2): # Fewer restarts for 39 as it's easier
            if time.time() > deadline:
                break
            x0 = np.random.randn(n * m * target_rank_39 + m * p * target_rank_39 + p * n * target_rank_39)
            try:
                res = least_squares(residuals_opt_39, x0, method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12)
                if res.success or res.cost < 1e-10:
                    U, V, W = unpack_params_39(res.x)
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    S_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                    if np.allclose(S_snap, M, atol=1e-9):
                        best_rank = 39
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        break
            except Exception:
                pass

    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
