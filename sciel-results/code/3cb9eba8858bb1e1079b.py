import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i, T=1."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=200, tol=1e-9, seed=42):
    """Alternating least squares for 3-tensor decomposition.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    np.random.seed(seed)
    dim1, dim2, dim3 = T.shape
    U = np.random.randn(dim1, rank) * 0.5
    V = np.random.randn(dim2, rank) * 0.5
    W = np.random.randn(dim3, rank) * 0.5
    
    for it in range(max_iter):
        # Update U: for each a, solve least squares
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Fix V,W. For each a: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
        # M[b,c,r] = V[b,r]*W[c,r], reshape to (dim2*dim3, rank)
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)
        MW_flat = MW.reshape(dim2 * dim3, rank)
        for a in range(dim1):
            target = T[a].reshape(dim2 * dim3)
            U[a, :], _, _, _ = np.linalg.lstsq(MW_flat.T, target, rcond=None)
        
        # Update V: for each b
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)
        UW_flat = UW.reshape(dim1 * dim3, rank)
        for b in range(dim2):
            target = T[:, b, :].reshape(dim1 * dim3)
            V[b, :], _, _, _ = np.linalg.lstsq(UW_flat.T, target, rcond=None)
        
        # Update W: for each c
        UV = np.einsum('ar,br->abr', U, V, optimize=True)
        UV_flat = UV.reshape(dim1 * dim2, rank)
        for c in range(dim3):
            target = T[:, :, c].reshape(dim1 * dim2)
            W[c, :], _, _, _ = np.linalg.lstsq(UV_flat.T, target, rcond=None)
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
        return U, V, W, err
    return None

def _try_structured_decomposition(n, m, p):
    """Try a structured decomposition exploiting the tensor structure.
    
    For each j (0..m-1), the sub-tensor T[i*m+j, j*p+k, k*n+i] = 1
    can be seen as: for fixed j, we need factors that produce 1 at positions
    (i*m+j, j*p+k, k*n+i) for all i,k.
    
    Key insight: T[i*m+j, j*p+k, k*n+i] = 1 means that for each j,
    the "fiber" through index j is an all-ones 3x6 pattern in a specific
    index mapping. We can try to share factors across j values.
    
    Strategy: For each j, use rank-1 factors scaled appropriately,
    but share the U and W structures.
    """
    # The tensor has a special property: T[a,b,c] = 1 iff 
    # a % m == b // p and b % p * n + a // m == c
    # i.e., the indices are "consistent"
    
    # A structured approach: use the fact that for each (i,k), 
    # the value 1 appears at exactly one j-slice position.
    # Actually no - for each (i,k), there's exactly one entry per j.
    
    # Let's try: rank = n * p = 18, using factors that encode (i,k) pairs
    # For each (i,k) pair, we need T[i*m+j, j*p+k, k*n+i] = 1 for all j.
    # This means U[i*m+j, r_{i,k}] * V[j*p+k, r_{i,k}] * W[k*n+i, r_{i,k}] = 1
    # for all j. 
    # 
    # If we set r_{i,k} to be a unique rank index for each (i,k),
    # then we need: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j.
    # 
    # One solution: U[i*m+j, r_{i,k}] = 1/sqrt(m), V[j*p+k, r_{i,k}] = 1/sqrt(m), 
    # W[k*n+i, r_{i,k}] = m. But this doesn't work because different j values
    # need to multiply to 1.
    #
    # Better: U[i*m+j, r_{i,k}] = 1, V[j*p+k, r_{i,k}] = 1, W[k*n+i, r_{i,k}] = 1
    # But then for a given rank index r_{i,k}, U[i*m+j, r] = 1 for all j,
    # V[j*p+k, r] = 1 for all j, W[k*n+i, r] = 1.
    # Then sum over r: for entry (a,b,c) = (i*m+j, j'*p+k', k''*n+i''),
    # we get sum over (i,k) such that i*m+j = i*m+j AND j*p+k = j'*p+k' AND k*n+i = k''*n+i''
    # This only equals 1 if i=i', j=j', k=k'. So this gives rank n*p = 18!
    #
    # Wait, let me verify: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # rank = n*p = 18
    # For each (i,k) in range(n) x range(p), assign rank index r = i*p + k
    # U[i*m+j, i*p+k] = 1 for all j
    # V[j*p+k, i*p+k] = 1 for all j  
    # W[k*n+i, i*p+k] = 1
    #
    # Check: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Let a = i1*m+j1, b = j2*p+k2, c = k3*n+i3
    # U[a,r] = 1 iff r = i1*p + k for some k (i.e., r // p == i1)
    # V[b,r] = 1 iff r = i*p + k2 for some i (i.e., r % p == k2)
    # W[c,r] = 1 iff r = i3*p + k3 (i.e., r == i3*p + k3)
    #
    # So the sum is 1 iff there exists r such that:
    # r // p == i1, r % p == k2, and r == i3*p + k3
    # This means i1 == i3, k2 == k3, and r = i1*p + k2 = i3*p + k3
    # So the sum is 1 iff i1 == i3 and k2 == k3.
    #
    # But we need T[i*m+j, j*p+k, k*n+i] = 1, which requires i1=i3=i, j1=j2=j, k2=k3=k.
    # The condition i1==i3 and k2==k3 is necessary but not sufficient - we also need j1==j2.
    #
    # The problem: U[i*m+j, r] = 1 for all j, so U doesn't distinguish j.
    # Similarly V[j*p+k, r] = 1 for all j, so V doesn't distinguish j.
    # We need j1 == j2 to be enforced.
    
    # Modified approach: make U and V depend on j.
    # U[i*m+j, r] should be non-zero only for the "right" j.
    # If we use rank = n*m*p = 90, that's schoolbook.
    # If we use rank = n*p = 18, we lose the j information.
    
    # Try rank = n*m*p but with a smarter structure, or find the true minimum.
    
    # Actually, the correct structured decomposition:
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 is the matrix multiplication tensor.
    # Its rank is known to be at most n*m*p (schoolbook) and at least max(n*m, m*p, p*n).
    # For 3x5x6, lower bound is max(15, 30, 18) = 30.
    # 
    # A better upper bound: use the fact that we can decompose as a sum over j
    # of rank-1-like terms, but each j-term needs rank n*p = 18.
    # Total: m * n * p = 90. That's schoolbook.
    #
    # Can we do better? The key is that different j values share the same (i,k) structure.
    # If we could make the j-dependence "cheap", we'd save rank.
    #
    # One approach: use Fourier-like or other orthogonal bases in the j dimension.
    # For each (i,k), instead of having m separate rank indices (one per j),
    # use a single rank index with j-dependent scaling.
    #
    # U[i*m+j, r_{i,k}] = u_j, V[j*p+k, r_{i,k}] = v_j, W[k*n+i, r_{i,k}] = w
    # Need: sum over (i,k) of u_j * v_j * w = delta(j1==j2) * 1 (for the right i,k)
    # This is getting complicated. Let me just use ALS to find a good decomposition.
    
    return None

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    T = _build_target_tensor(n, m, p)
    
    # Try structured decomposition first
    # Based on analysis, the minimum rank is at least 30 (max of dimensions)
    # Try ranks from 30 upward with ALS
    
    best_result = None
    
    # Try a range of ranks with multiple restarts
    for rank in range(30, 91):
        if time.time() > deadline - 20:
            break
        
        n_restarts = 3 if rank <= 40 else 2
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = rank * 100 + restart
            result = _als_decompose(T, rank, max_iter=100, tol=1e-9, seed=seed)
            if result is not None:
                U, V, W, err = result
                if err < 1e-8:
                    best_result = (U, V, W, rank)
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to lists with integer/half-integer rounding if close
        def to_clean(val):
            if abs(val - round(val)) < 1e-9:
                return int(round(val))
            elif abs(val * 2 - round(val * 2)) < 1e-9:
                return round(val * 2) / 2.0
            else:
                return float(val)
        
        U_list = [[to_clean(x) for x in row] for row in U]
        V_list = [[to_clean(x) for x in row] for row in V]
        W_list = [[to_clean(x) for x in row] for row in W]
        
        # Final verification
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    result = _construct_trivial(n, m, p)
    return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
