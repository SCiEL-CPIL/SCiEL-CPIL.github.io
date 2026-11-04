"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import sys
    sys.setrecursionlimit(10000)
    
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the tensor as a flat dictionary of non-zero entries for efficiency
    # Key: (a, b, c), Value: Fraction
    tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[(a, b, c)] = Fraction(1)
    
    U = []
    V = []
    W = []
    
    # Helper to count non-zeros in a dict
    def count_nonzeros(d):
        return sum(1 for v in d.values() if v != 0)
    
    # Helper to subtract a rank-1 tensor defined by u, v, w
    def subtract_rank1(d, u, v, w):
        # We need to update the dictionary
        # New value at (a,b,c) is d[(a,b,c)] - u[a]*v[b]*w[c]
        # Since u, v, w are sparse, we can iterate over their supports
        # But the rank-1 tensor is dense if u,v,w are dense.
        # In our greedy, u,v,w are derived from the current non-zeros, so they are sparse-ish.
        
        # To be safe and exact, we iterate over all existing keys and potentially new keys?
        # No, the support of the new tensor is the union of the old support and the support of the rank-1 tensor.
        # However, the rank-1 tensor might have non-zeros where d is 0.
        
        # Optimization: Since we are looking for minimal rank, and the tensor is small,
        # we can just update the existing keys and check if any new keys become non-zero.
        # But wait, if d is 0 at (a,b,c) and u[a]v[b]w[c] != 0, then the new value is non-zero.
        # So we must consider the support of u, v, w.
        
        # Let's collect all relevant indices.
        # Actually, for this specific problem, the "greedy" rank-1 tensor constructed
        # from an anchor (a0,b0,c0) has:
        # u[a] non-zero only if d[a,b0,c0] was non-zero (or a==a0)
        # v[b] non-zero only if d[a0,b,c0] was non-zero (or b==b0)
        # w[c] non-zero only if d[a0,b0,c] was non-zero (or c==c0)
        
        # So the support of the rank-1 tensor is bounded by the "cross" of the anchor in the current tensor.
        
        # Let's just do the full update for simplicity, as dims are small (8, 20, 10).
        # Max 1600 entries.
        
        new_d = dict(d)
        
        # Iterate over all possible (a,b,c) that could be affected?
        # No, only those where u[a]!=0, v[b]!=0, w[c]!=0.
        
        # Get supports
        sup_u = [a for a in range(dim_u) if u[a] != 0]
        sup_v = [b for b in range(dim_v) if v[b] != 0]
        sup_w = [c for c in range(dim_w) if w[c] != 0]
        
        # We need to update keys in d that are in the product of supports,
        # AND any existing keys in d that might become zero?
        # Actually, we just need to update the dictionary.
        
        # 1. Update existing keys
        for key, val in list(d.items()):
            a, b, c = key
            val_new = val - u[a] * v[b] * w[c]
            if val_new == 0:
                del new_d[key]
            else:
                new_d[key] = val_new
        
        # 2. Check for new non-zeros created by the subtraction
        # If d[key] was 0 (not in d), new val is -u[a]v[b]w[c].
        # This is non-zero if u[a]!=0, v[b]!=0, w[c]!=0.
        # We only need to add these if they weren't in d.
        
        for a in sup_u:
            for b in sup_v:
                for c in sup_w:
                    key = (a, b, c)
                    if key not in d: # It was zero
                        val_new = -u[a] * v[b] * w[c]
                        if val_new != 0:
                            new_d[key] = val_new
                    # If key was in d, it was already handled in step 1.
                    # Note: Step 1 iterated over d.items(). If key was in d, it's updated.
                    # If key was NOT in d, it's 0. Subtraction gives -prod.
        
        return new_d
    
    # Greedy Loop
    max_iter = 100
    for _ in range(max_iter):
        if not tensor:
            break
        
        # Find the best anchor
        # Candidate anchors are all non-zero entries.
        # For each, simulate the extraction.
        # To save time, we can limit the candidates if the tensor is large,
        # but here it's small.
        
        best_anchor = None
        best_new_tensor = None
        min_nonzeros = float('inf')
        
        # Optimization: Only check a subset if it's too slow?
        # Let's check all.
        
        # To speed up, we can precompute the "cross" supports for each anchor?
        # No, just loop.
        
        # List of non-zero keys
        keys = list(tensor.keys())
        
        for key in keys:
            a0, b0, c0 = key
            val0 = tensor[key]
            if val0 == 0: continue
            
            # Construct u, v, w
            u = [Fraction(0)] * dim_u
            v = [Fraction(0)] * dim_v
            w = [Fraction(0)] * dim_w
            
            u[a0] = Fraction(1)
            v[b0] = Fraction(1)
            w[c0] = Fraction(1)
            
            # Fill u based on column (b0, c0)
            for a in range(dim_u):
                if a == a0: continue
                k = (a, b0, c0)
                if k in tensor and tensor[k] != 0:
                    u[a] = tensor[k] / (v[b0] * w[c0])
            
            # Fill v based on row (a0, c0)
            for b in range(dim_v):
                if b == b0: continue
                k = (a0, b, c0)
                if k in tensor and tensor[k] != 0:
                    v[b] = tensor[k] / (u[a0] * w[c0])
            
            # Fill w based on depth (a0, b0)
            for c in range(dim_w):
                if c == c0: continue
                k = (a0, b0, c)
                if k in tensor and tensor[k] != 0:
                    w[c] = tensor[k] / (u[a0] * v[b0])
            
            # Subtract and count
            new_t = subtract_rank1(tensor, u, v, w)
            cnt = len(new_t)
            
            if cnt < min_nonzeros:
                min_nonzeros = cnt
                best_anchor = (a0, b0, c0)
                best_new_tensor = new_t
                # Store u, v, w for this best anchor?
                # We can re-construct them later, or store them.
                # Storing is safer.
                best_u, best_v, best_w = u, v, w
        
        if best_anchor is None:
            break
            
        # Apply the best extraction
        tensor = best_new_tensor
        U.append(best_u)
        V.append(best_v)
        W.append(best_w)
        
        # If we have extracted more than the schoolbook rank, stop (shouldn't happen)
        if len(U) >= n * m * p:
            break
    
    # Convert to output format
    # Ensure we return floats or ints
    def convert_list(lst):
        return [float(x) if x.denominator == 1 else float(x) for x in lst]
    
    if len(U) < n * m * p:
        return {
            "U": [convert_list(row) for row in U],
            "V": [convert_list(row) for row in V],
            "W": [convert_list(row) for row in W]
        }
    else:
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
