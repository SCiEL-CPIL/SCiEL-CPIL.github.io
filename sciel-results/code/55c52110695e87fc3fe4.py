import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor: all 1s. Scale by 2 to work with integers.
    # Target entries are 2.
    target = np.ones((n, m, p), dtype=int) * 2
    target_flat = target.flatten()
    
    # Grid for vector entries
    # We use 0..4 to keep numbers small.
    # If we need negative numbers, we can expand, but positive sums are easier to manage for "all positive" target.
    # However, cancellations might be needed.
    # Let's stick to 0..4 first. If it fails, we might need negative.
    # But for "all 2s", positive tensors summing to 2s is a valid path.
    # e.g. 1+1=2.
    
    # Pre-compute candidates
    # u: n, v: m, w: p
    # n=2, m=3, p=4
    
    # To save memory and time, we might generate on the fly if the count is high.
    # Count: 5^2 * 5^3 * 5^4 = 25 * 125 * 625 = 1,953,125.
    # This is ~2M.
    # Storing 2M vectors of length 24 (int8) is 48MB. OK.
    # Storing norms (float) is 16MB. OK.
    
    # However, generating 2M tensors and flattening them in Python loops is slow.
    # We can use numpy broadcasting to generate them?
    # Or just loop.
    # 2M iterations in Python might take 10-20s just to generate.
    # Then 18 * 2M dot products.
    # Total might exceed 120s.
    
    # Optimization:
    # We can restrict the search space.
    # Notice that the target is constant.
    # We can use a smaller grid, e.g., 0..2.
    # 3^2 * 3^3 * 3^4 = 9 * 27 * 81 = 19,683.
    # This is very small.
    # Let's try grid 0..2 first. If it finds a solution, great.
    # If not, try 0..3.
    # 4^2 * 4^3 * 4^4 = 16 * 64 * 256 = 262,144.
    # This is also small.
    
    max_val = 2
    # If max_val=2 fails, try 3.
    
    def try_decomposition(max_val, target_rank):
        # Generate candidates
        u_vals = np.arange(max_val + 1)
        v_vals = np.arange(max_val + 1)
        w_vals = np.arange(max_val + 1)
        
        # We need all combinations.
        # This is a Cartesian product.
        # Using itertools.product is slow for 2M.
        # But for 262k, it's fine.
        
        import itertools
        
        candidates = []
        norms = []
        
        # Iterate over all u, v, w
        # To speed up, we can pre-compute flattened u, v, w? No, tensor is outer product.
        
        # Let's generate in a loop.
        # For max_val=3, count is 262k.
        # For max_val=4, count is 1.9M.
        
        for u_tuple in itertools.product(u_vals, repeat=n):
            u_vec = np.array(u_tuple)
            if np.all(u_vec == 0): continue
            for v_tuple in itertools.product(v_vals, repeat=m):
                v_vec = np.array(v_tuple)
                if np.all(v_vec == 0): continue
                for w_tuple in itertools.product(w_vals, repeat=p):
                    w_vec = np.array(w_tuple)
                    if np.all(w_vec == 0): continue
                    
                    # Compute tensor
                    t = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
                    t_flat = t.flatten()
                    norm_sq = np.dot(t_flat, t_flat)
                    
                    candidates.append(t_flat)
                    norms.append(norm_sq)
                    # Store u, v, w for reconstruction
                    # We can just store the tuples
                    # But we need to return U, V, W matrices.
                    # So we need to keep track of which u, v, w were used.
                    
        # This is getting complicated to store.
        # Let's store indices or the vectors themselves.
        
        # Better approach:
        # Pre-compute a list of (u_vec, v_vec, w_vec, t_flat, norm_sq)
        # This is memory heavy but OK for 262k.
        
        pass

    # Refined Plan:
    # 1. Try max_val=2, ranks 18, 19, 20.
    # 2. Try max_val=3, ranks 18, 19, 20.
    # 3. Fallback to rank 24.
    
    # Implementation of greedy search
    
    def run_greedy(max_val, target_rank):
        import itertools
        import numpy as np
        
        u_vals = np.arange(max_val + 1)
        v_vals = np.arange(max_val + 1)
        w_vals = np.arange(max_val + 1)
        
        # Pre-compute candidates
        # To save memory, we can store the vectors and compute t on the fly?
        # No, computing t (24 mults) is cheap.
        # But we need to do it for every candidate in every step.
        # So pre-computing t_flat is better.
        
        cand_u = []
        cand_v = []
        cand_w = []
        cand_t = []
        cand_norm = []
        
        # Optimization: Only generate non-zero vectors
        # And maybe filter out "too small" or "too large"?
        
        for u_tuple in itertools.product(u_vals, repeat=n):
            u_vec = np.array(u_tuple, dtype=float)
            if np.all(u_vec == 0): continue
            for v_tuple in itertools.product(v_vals, repeat=m):
                v_vec = np.array(v_tuple, dtype=float)
                if np.all(v_vec == 0): continue
                for w_tuple in itertools.product(w_vals, repeat=p):
                    w_vec = np.array(w_tuple, dtype=float)
                    if np.all(w_vec == 0): continue
                    
                    t = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
                    t_flat = t.flatten()
                    norm_sq = float(np.dot(t_flat, t_flat))
                    
                    cand_u.append(u_vec)
                    cand_v.append(v_vec)
                    cand_w.append(w_vec)
                    cand_t.append(t_flat)
                    cand_norm.append(norm_sq)
                    
        # Convert to numpy arrays for fast dot product
        # cand_t is a list of arrays. Stack them.
        T_cand = np.array(cand_t) # Shape (N, 24)
        N_norms = np.array(cand_norm)
        
        T_rem = target_flat.copy()
        selected_idx = []
        
        for step in range(target_rank):
            # Minimize || T_rem - t ||^2
            # = ||T_rem||^2 - 2 <T_rem, t> + ||t||^2
            # ||T_rem||^2 is constant.
            # Minimize -2 <T_rem, t> + ||t||^2
            # Equivalent to maximizing 2 <T_rem, t> - ||t||^2
            
            # Compute dot products: T_cand @ T_rem
            # Shape (N, 24) @ (24,) -> (N,)
            dots = T_cand.dot(T_rem)
            
            # Score = 2 * dots - N_norms
            scores = 2 * dots - N_norms
            
            # Find best index
            best_idx = np.argmax(scores)
            
            # Update T_rem
            T_rem = T_rem - T_cand[best_idx]
            selected_idx.append(best_idx)
            
            # Check if T_rem is close to 0
            if np.allclose(T_rem, 0):
                break
                
        # Verify
        if not np.allclose(T_rem, 0):
            return None
            
        # Construct U, V, W
        U = np.zeros((n*m, target_rank), dtype=float)
        V = np.zeros((m*p, target_rank), dtype=float)
        W = np.zeros((p*n, target_rank), dtype=float)
        
        for r, idx in enumerate(selected_idx):
            u_vec = cand_u[idx]
            v_vec = cand_v[idx]
            w_vec = cand_w[idx]
            
            # Map indices
            # U[i*m+j, r] = u_vec[i] * v_vec[j] ? 
            # No, the decomposition is sum_r U[:,r] V[:,r] W[:,r].
            # U[:,r] is a vector of length n*m.
            # V[:,r] is a vector of length m*p.
            # W[:,r] is a vector of length p*n.
            
            # Our tensor term is u_vec[i] * v_vec[j] * w_vec[k].
            # We need to distribute these into U, V, W.
            # We can put all of u in U, all of v in V, all of w in W.
            # U[i*m+j, r] = u_vec[i] ? No, U depends on i and j.
            # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
            # We want this to equal u_vec[i] * v_vec[j] * w_vec[k].
            
            # We can set:
            # U[i*m+j, r] = u_vec[i]
            # V[j*p+k, r] = v_vec[j]
            # W[k*n+i, r] = w_vec[k]
            
            # Check:
            # Sum over r: U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
            # = sum_r u_r[i] v_r[j] w_r[k]
            # This matches the tensor definition.
            
            for i in range(n):
                for j in range(m):
                    U[i*m + j, r] = u_vec[i]
                    
            for j in range(m):
                for k in range(p):
                    V[j*p + k, r] = v_vec[j]
                    
            for k in range(p):
                for i in range(n):
                    W[k*n + i, r] = w_vec[k]
                    
        return U, V, W

    # Try different max_vals and ranks
    # Note: The problem asks for integer/half-integer.
    # Our vectors are integers. So U, V, W entries are integers.
    # This is valid.
    
    for max_val in [2, 3, 4]:
        for rank in [18, 19, 20]:
            res = run_greedy(max_val, rank)
            if res:
                U, V, W = res
                # Convert to list of lists as required?
                # "factor matrices of shapes ... with integer/half-integer entries"
                # Return type is dict of matrices.
                # The example returns lists of lists.
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook
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
