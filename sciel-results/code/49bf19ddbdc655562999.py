<<<<<<< SEARCH
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    return _schoolbook_decomposition(n, m, p)
=======
def _verify_decomposition_fast(U, V, W, n, m, p, rank):
    """Vectorized verification using numpy."""
    import numpy as np
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    # Reshape: U (n*m, rank) -> (n, m, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Compute via einsum: first U*V gives (n, p, rank)
    UV = np.einsum('ijr,jkr->ipr', U3, V3)  # (n, p, rank)
    T = np.einsum('ipr,kir->ikp', UV, W3)   # (n, p, p) -> wait need (i,j,k)
    # Actually: T[i,j,k] where j is from U, k from V
    # U3[i,j,r], V3[j,k,r], W3[k,i,r]
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = sum_r (U3[i,j,r] * V3[j,k,r]) * W3[k,i,r]
    # Let's just do it directly with broadcasting
    # U3[i,j,r] * V3[j,k,r] * W3[k,i,r] summed over r
    # Shape: (n, m, rank) * (m, p, rank) * (p, n, rank)
    # Need to align: U3[i,j,r], V3[j,k,r], W3[k,i,r]
    # Result: (n, m, p)
    T = np.einsum('ijr,jkr,kir->ij k', U3, V3, W3)
    # Actually let me be more careful with einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.all(T == 1)


def _try_merge_slices(U, V, W, r1, r2, n, m, p):
    """
    Try to merge rank slices r1 and r2 into a single rank-1 term.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    """
    import numpy as np
    rank = len(U[0])
    
    # The contribution of slices r1 and r2 to the tensor:
    # For each (i,j,k): delta[i,j,k] = U[i,j,r1]*V[j,k,r1]*W[k,i,r1] + U[i,j,r2]*V[j,k,r2]*W[k,i,r2]
    # We need: delta[i,j,k] = c1*U'[i,j]*V'[j,k]*W'[k,i] for some integer vectors
    # 
    # Strategy: try to find U', V', W' such that:
    #   U'[i,j] * V'[j,k] * W'[k,i] = delta[i,j,k] / c1
    #
    # This is a rank-1 tensor factorization problem.
    # 
    # Approach: Pick a reference entry, say (i0,j0,k0) where delta != 0.
    # Then c1 * U'[i0,j0] * V'[j0,k0] * W'[k0,i0] = delta[i0,j0,k0]
    # 
    # We can fix U'[i0,j0] = 1 (absorbing into c1), then:
    #   c1 * V'[j0,k0] * W'[k0,i0] = delta[i0,j0,k0]
    # 
    # This is still underdetermined. Let's use a different approach:
    # 
    # For a rank-1 tensor T[i,j,k] = a[i,j]*b[j,k]*c[k,i], we can recover:
    #   Fix i=0, j=0: T[0,0,k] = a[0,0]*b[0,k]*c[k,0] for all k
    #   Fix i=0, k=0: T[0,j,0] = a[0,j]*b[j,0]*c[0,0] for all j
    #   Fix j=0, k=0: T[i,0,0] = a[i,0]*b[0,0]*c[0,i] for all i
    #
    # Let's try: assume the merged tensor is rank-1. Check if it is.
    
    # Build the delta tensor
    delta = np.zeros((n, m, p), dtype=np.int64)
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    
    for r in [r1, r2]:
        delta += Ua[:, r].reshape(n, m)[:, :, np.newaxis] * Va[:, r].reshape(m, p)[np.newaxis, :, :] * Wa[:, r].reshape(p, n)[np.newaxis, :, :]
    
    # Check if delta is rank-1: delta[i,j,k] = a[i,j]*b[j,k]*c[k,i]
    # Method: try to factor it
    # Find a nonzero entry
    nz = np.nonzero(delta)
    if len(nz[0]) == 0:
        # Both slices contribute zero - just remove one
        return True, U, V, W  # degenerate case
    
    # Try to factor: for each (i,j), delta[i,j,:] should be proportional to b[j,k]*c[k,i]
    # And for fixed (i,k), delta[i,:,k] should be proportional to a[i,j]*b[j,k]
    
    # Pick reference: i0, j0, k0
    i0, j0, k0 = int(nz[0][0]), int(nz[1][0]), int(nz[2][0])
    
    # Try: set a[i0,j0] = 1, then b[j0,k]*c[k,i0] = delta[i0,j0,k] for all k
    # And a[i,j0]*b[j0,k]*c[k,i] = delta[i,j0,k] => a[i,j0] = delta[i,j0,k] / (b[j0,k]*c[k,i])
    # This is getting complex. Let's use a simpler check:
    # 
    # A tensor T is rank-1 (in this specific factorization form) iff:
    # T[i,j,k]*T[i',j',k'] * T[i',j,k] * T[i,j',k'] == T[i,j,k']*T[i',j',k] * T[i',j,k']*T[i,j',k]
    # (Plücker-type relation)
    #
    # Simpler: check if all 2x2 minors of the "unfolded" tensors are zero.
    # 
    # Actually, the simplest approach: just try to recover a, b, c directly.
    # 
    # From T[i,j,k] = a[i,j]*b[j,k]*c[k,i]:
    # T[i,j,k] * T[i,j',k'] = a[i,j]*b[j,k]*c[k,i]*a[i,j']*b[j',k']*c[k',i']
    # T[i,j,k'] * T[i,j',k] = a[i,j]*b[j,k']*c[k',i]*a[i,j']*b[j',k]*c[k,i]
    # Ratio: T[i,j,k]*T[i,j',k'] / (T[i,j,k']*T[i,j',k]) = b[j,k]*b[j',k'] / (b[j,k']*b[j',k])
    #
    # Let's just try a direct factorization approach:
    # Fix j=0: T[i,0,k] = a[i,0]*b[0,k]*c[k,i]
    # Fix i=0: T[0,j,k] = a[0,j]*b[j,k]*c[k,0]
    # Fix k=0: T[i,j,0] = a[i,j]*b[j,0]*c[0,i]
    
    # Let's try: assume b[0,k] = delta[0,0,k] / (a[0,0]*c[k,0])
    # And a[0,j] = delta[0,j,0] / (b[j,0]*c[0,0])
    # This is circular. Let's use a numerical approach with numpy.
    
    # Alternative: use the fact that for rank-1, we can write:
    # delta[i,j,k] = (delta[i,j,0] * delta[0,0,k]) / delta[0,j,0] * (delta[i,0,0] * delta[0,j,k]) / delta[0,0,0]
    # ... this is getting too complex.
    
    # Let's use a practical approach: try small integer values for the new slice.
    # The new slice must satisfy:
    #   sum over (i,j,k) of (delta[i,j,k] - new_U[i,j]*new_V[j,k]*new_W[k,i])^2 = 0
    #
    # Since we're looking for exact integer solutions, let's try:
    # 1. Extract the "profile" of the delta tensor
    # 2. Try to factor it as a rank-1 tensor
    
    # Factorization attempt:
    # delta[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    # 
    # Step 1: Fix i=0, j=0: delta[0,0,k] = a[0,0] * b[0,k] * c[k,0] for all k
    # Step 2: Fix i=0, k=0: delta[0,j,0] = a[0,j] * b[j,0] * c[0,0] for all j  
    # Step 3: Fix j=0, k=0: delta[i,0,0] = a[i,0] * b[0,0] * c[0,i] for all i
    
    # From step 1: b[0,k]*c[k,0] = delta[0,0,k]/a[0,0]
    # From step 2: a[0,j]*b[j,0] = delta[0,j,0]/c[0,0]
    # From step 3: a[i,0]*c[0,i] = delta[i,0,0]/b[0,0]
    
    # Let's try a[0,0] = 1, c[0,0] = 1, b[0,0] = 1 (normalize)
    # Then: b[0,k]*c[k,0] = delta[0,0,k]
    #        a[0,j]*b[j,0] = delta[0,j,0]
    #        a[i,0]*c[0,i] = delta[i,0,0]
    
    # This is still underdetermined. Let's just check if the tensor has rank 1
    # by verifying the Plücker relation for a few random quadruples.
    
    # Actually, let me just check: is delta a rank-1 tensor in the standard sense?
    # Flatten to matrix and check rank.
    # delta[i,j,k] as matrix M[i,j;k] (n*m x p) - rank should be 1
    M1 = delta.reshape(n*m, p)
    M2 = delta.reshape(n, m*p)
    M3 = delta.reshape(m, n*p)
    
    # Check if all three unfoldings have rank <= 1
    from numpy.linalg import matrix_rank
    if matrix_rank(M1.astype(float)) > 1 or matrix_rank(M2.astype(float)) > 1 or matrix_rank(M3.astype(float)) > 1:
        return False, None, None, None
    
    # It's rank-1! Now factor it.
    # delta[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    # Use SVD or direct extraction.
    
    # From M1 = delta.reshape(n*m, p): M1[i*m+j, k] = a[i,j]*b[j,k]*c[k,i]
    # This isn't a simple outer product in general.
    
    # Let me try a different factorization:
    # Pick i0=0, j0=0: delta[0,0,k] = a[0,0]*b[0,k]*c[k,0]
    # If a[0,0] != 0: b[0,k]*c[k,0] = delta[0,0,k]/a[0,0]
    
    # This is getting too complex for a general solution. Let me use a simpler heuristic:
    # Try to see if we can express delta as a single outer product pattern.
    
    # For now, return False - the merge attempt failed.
    return False, None, None, None


def _search_rank_48(U, V, W, n, m, p, deadline):
    """
    Search for a rank-48 decomposition by trying to merge pairs of rank slices.
    Start from the rank-49 Strassen decomposition and try to eliminate one slice.
    """
    import numpy as np
    import random
    
    rank = len(U[0])
    if rank != 49:
        return None
    
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    
    # For each pair (r1, r2), check if their combined contribution is rank-1
    # delta[i,j,k] = U[i,j,r1]*V[j,k,r1]*W[k,i,r1] + U[i,j,r2]*V[j,k,r2]*W[k,i,r2]
    
    # Precompute the contribution of each slice
    slice_contrib = []
    for r in range(rank):
        contrib = Ua[:, r].reshape(n, m)[:, :, np.newaxis] * Va[:, r].reshape(m, p)[np.newaxis, :, :] * Wa[:, r].reshape(p, n)[np.newaxis, :, :]
        slice_contrib.append(contrib)
    
    # Try all pairs
    pairs = list(range(rank))
    random.seed(42)
    random.shuffle(pairs)
    
    # Try a limited number of pairs to stay within time budget
    max_pairs_to_try = 200
    
    for idx in range(min(len(pairs), max_pairs_to_try)):
        if time.time() > deadline - 20:
            break
        
        r1 = pairs[idx]
        # Try merging with each other slice
        for r2 in range(rank):
            if r2 == r1:
                continue
            if time.time() > deadline - 20:
                break
            
            # Combined contribution
            delta = slice_contrib[r1] + slice_contrib[r2]
            
            # Check if delta is rank-1 in the tensor sense
            # delta[i,j,k] = a[i,j]*b[j,k]*c[k,i]
            
            # Quick check: all entries should be consistent with a rank-1 factorization
            # delta[i,j,k] * delta[i',j',k'] should equal delta[i,j',k'] * delta[i',j,k] * (something)
            # Actually for the specific form a[i,j]*b[j,k]*c[k,i]:
            # delta[i,j,k] * delta[i',j',k'] = a[i,j]*b[j,k]*c[k,i]*a[i',j']*b[j',k']*c[k',i']
            # delta[i',j,k'] * delta[i,j',k] = a[i',j]*b[j,k']*c[k',i']*a[i,j']*b[j',k]*c[k,i]
            # These aren't simply related.
            
            # Simpler check: use the unfolding rank test
            M1 = delta.reshape(n*m, p).astype(float)
            from numpy.linalg import matrix_rank
            if matrix_rank(M1) > 1:
                continue
            
            # Try to factor: delta[i,j,k] = a[i,j]*b[j,k]*c[k,i]
            # Method: 
            #   delta[i,j,0] = a[i,j]*b[j,0]*c[0,i]  ... (1)
            #   delta[i,0,k] = a[i,0]*b[0,k]*c[k,i]  ... (2)
            #   delta[0,j,k] = a[0,j]*b[j,k]*c[k,0]  ... (3)
            
            # From (1) and (2): delta[i,j,0]*delta[i,0,k] = a[i,j]*b[j,0]*c[0,i]*a[i,0]*b[0,k]*c[k,i]
            # From (3) with i->i, j->j, k->0: delta[0,j,0] = a[0,j]*b[j,0]*c[0,0]
            
            # Let's try a direct extraction:
            # Assume c[0,i] = 1 for all i (normalize), then:
            #   delta[i,j,0] = a[i,j]*b[j,0]
            #   delta[i,0,k] = a[i,0]*b[0,k]*c[k,i]
            #   delta[0,j,k] = a[0,j]*b[j,k]*c[k,0]
            
            # From delta[i,j,0] = a[i,j]*b[j,0]:
            #   a[i,j] = delta[i,j,0] / b[j,0]
            #   So delta[i,j,0] / delta[i',j,0] = a[i,j]/a[i',j] (independent of j if b[j,0] cancels)
            #   Actually: delta[i,j,0]/delta[i,0,0] = a[i,j]*b[j,0]/(a[i,0]*b[0,0])
            
            # This is getting complex. Let me try a numerical approach:
            # Use the fact that for rank-1, we can write delta as a product of three vectors
            # in appropriate spaces.
            
            # Try: a[i,j] = delta[i,j,0] / b[j,0], and we need to find b[j,0] and c[0,i]
            # such that everything is consistent.
            
            # Let's try: set b[j,0] = 1 for all j (normalize)
            # Then a[i,j] = delta[i,j,0]
            # delta[i,0,k] = a[i,0]*b[0,k]*c[k,i] = delta[i,0,0]*b[0,k]*c[k,i]
            # delta[0,j,k] = a[0,j]*b[j,k]*c[k,0] = delta[0,j,0]*b[j,k]*c[k,0]
            # delta[i,j,k] = a[i,j]*b[j,k]*c[k,i] = delta[i,j,0]*b[j,k]*c[k,i]
            
            # From delta[i,0,k] = delta[i,0,0]*b[0,k]*c[k,i]:
            #   b[0,k]*c[k,i] = delta[i,0,k]/delta[i,0,0] (for delta[i,0,0] != 0)
            # This must be independent of i!
            
            # Check: for k=0: b[0,0]*c[0,i] = delta[i,0,0]/delta[i,0,0] = 1
            # So b[0,0]*c[0,i] = 1 for all i.
            
            # For general k: b[0,k]*c[k,i] = delta[i,0,k]/delta[i,0,0]
            # This must be independent of i.
            
            valid = True
            for k in range(p):
                if k == 0:
                    continue
                ratios = []
                for i in range(n):
                    if delta[i,0,0] == 0:
                        if delta[i,0,k] != 0:
                            valid = False
                            break
                        continue
                    ratios.append(delta[i,0,k] / delta[i,0,0])
                if not valid:
                    break
                if len(ratios) > 1:
                    # All ratios must be equal
                    if any(abs(r - ratios[0]) > 1e-9 for r in ratios[1:]):
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Similarly check for the j dimension
            for j in range(m):
                if j == 0:
                    continue
                ratios = []
                for k in range(p):
                    if delta[0,j,0] == 0:
                        if delta[0,j,k] != 0:
                            valid = False
                            break
                        continue
                    ratios.append(delta[0,j,k] / delta[0,j,0])
                if not valid:
                    break
                if len(ratios) > 1:
                    if any(abs(r - ratios[0]) > 1e-9 for r in ratios[1:]):
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Now try to construct the factorization
            # a[i,j] = delta[i,j,0] (with b[j,0]=1)
            # b[0,k]*c[k,i] = delta[i,0,k]/delta[i,0,0]
            # b[j,k]*c[k,0] = delta[0,j,k]/delta[0,j,0]
            
            # Let's set c[k,0] = 1 for all k (another normalization)
            # Then b[j,k] = delta[0,j,k]/delta[0,j,0]
            # And b[0,k]*c[k,i] = delta[i,0,k]/delta[i,0,0]
            # => c[k,i] = delta[i,0,k]/(delta[i,0,0]*b[0,k])
            
            # b[0,k] = delta[0,0,k]/delta[0,0,0]
            # c[k,i] = delta[i,0,k]*delta[0,0,0]/(delta[i,0,0]*delta[0,0,k])
            
            # Check consistency: c[k,0] should be 1
            # c[k,0] = delta[0,0,k]*delta[0,0,0]/(delta[0,0,0]*delta[0,0,k]) = 1 ✓
            
            # Now verify: delta[i,j,k] should equal a[i,j]*b[j,k]*c[k,i]
            # = delta[i,j,0] * (delta[0,j,k]/delta[0,j,0]) * (delta[i,0,k]*delta[0,0,0]/(delta[i,0,0]*delta[0,0,k]))
            
            new_a = np.zeros((n, m), dtype=np.int64)
            new_b = np.zeros((m, p), dtype=np.int64)
            new_c = np.zeros((p, n), dtype=np.int64)
            
            for i in range(n):
                for j in range(m):
                    new_a[i,j] = delta[i,j,0]
            
            for j in range(m):
                for k in range(p):
                    if delta[0,j,0] != 0:
                        new_b[j,k] = delta[0,j,k] // delta[0,j,0] if delta[0,j,k] % delta[0,j,0] == 0 else 0
                    else:
                        new_b[j,k] = 0
            
            for k in range(p):
                for i in range(n):
                    if delta[i,0,0] != 0 and delta[0,0,k] != 0:
                        val = delta[i,0,k] * delta[0,0,0]
                        den = delta[i,0,0] * delta[0,0,k]
                        if den != 0 and val % den == 0:
                            new_c[k,i] = val // den
                        else:
                            new_c[k,i] = 0
                    else:
                        new_c[k,i] = 0
            
            # Verify the factorization
            reconstructed = new_a[:, :, np.newaxis] * new_b[np.newaxis, :, :] * new_c[np.newaxis, :, :]
            if np.array_equal(reconstructed, delta):
                # Success! Create new decomposition with rank 48
                new_rank = rank - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                # Copy all slices except r1, r2
                new_r = 0
                for r in range(rank):
                    if r == r1 or r == r2:
                        continue
                    for idx_u in range(n * m):
                        new_U[idx_u][new_r] = U[idx_u][r]
                    for idx_v in range(m * p):
                        new_V[idx_v][new_r] = V[idx_v][r]
                    for idx_w in range(p * n):
                        new_W[idx_w][new_r] = W[idx_w][r]
                    new_r += 1
                
                # Add the merged slice
                for i in range(n):
                    for j in range(m):
                        new_U[i*m+j][new_r] = int(new_a[i,j])
                for j in range(m):
                    for k in range(p):
                        new_V[j*p+k][new_r] = int(new_b[j,k])
                for k in range(p):
                    for i in range(n):
                        new_W[k*n+i][new_r] = int(new_c[k,i])
                
                # Verify
                if _verify_decomposition_fast(new_U, new_V, new_W, n, m, p, new_rank):
                    return {"U": new_U, "V": new_V, "W": new_W}
            
            # Try other normalizations
            # ... (additional attempts would go here)
    
    return None


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    if _verify_decomposition(U, V, W, n, m, p, rank):
        # Try to find a rank-48 decomposition via slice merging
        if time.time() < deadline - 30:
            rank48 = _search_rank_48(U, V, W, n, m, p, deadline)
            if rank48 is not None:
                return rank48
        return result

    return _schoolbook_decomposition(n, m, p)
>>>>>>> REPLACE
