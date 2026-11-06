import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    N = 24
    
    # --- High Precision Sup-Norm Evaluator ---
    # We use a dense grid and refine the peak if needed.
    # For the search, a good approximation is sufficient.
    # For the final answer, we return coefficients, and the evaluator computes the exact score.
    # So we just need to find the *best* coefficients.
    
    # Grid for fast evaluation during search
    M_search = 2048
    angles = np.linspace(0, 2 * np.pi, M_search, endpoint=False)
    # Z[m, k] = exp(i * m * theta_0 * k)
    # We can construct this using FFT or direct exp.
    # Direct exp is fine for 2048x24.
    k_vec = np.arange(N)
    Z = np.exp(1j * np.outer(angles, k_vec))
    
    def sup_norm_fast(coeffs):
        """
        Compute max |P(z)| on the search grid.
        coeffs: array of shape (N,) with values +/- 1
        """
        # coeffs @ Z.T gives values at each angle
        vals = coeffs @ Z.T
        return np.max(np.abs(vals))

    # --- Rudin-Shapiro Initial Guess ---
    def rudin_shapiro(length):
        p, q = [1], [1]
        while len(p) < length:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:length], dtype=np.float64)
    
    # --- Local Search ---
    def local_search(initial_coeffs, max_iters=1000):
        current = initial_coeffs.copy()
        current_val = sup_norm_fast(current)
        
        # Precompute Z columns for efficient update?
        # Flipping k: new_vals = old_vals - 2 * c_k * Z[:, k]
        # We can just recompute max abs for each flip candidate.
        
        for _ in range(max_iters):
            if time.time() > deadline - 20.0:
                break
                
            best_flip = -1
            best_new_val = current_val
            
            # Try flipping each coefficient
            # Optimization: 
            # vals = current @ Z.T
            # For each k:
            #   delta = -2 * current[k] * Z[:, k]
            #   new_vals = vals + delta
            #   new_norm = max(|new_vals|)
            
            base_vals = current @ Z.T
            
            # Vectorize the flip evaluation?
            # We can compute the effect of all flips at once?
            # new_vals[:, j] = base_vals[j] - 2 * current[k_j] * Z[:, k_j]
            # This is hard to vectorize into a single max operation directly
            # because the max is over the frequency domain (rows of Z), not the coefficients.
            
            # Let's just loop over N=24. It's fast.
            for k in range(N):
                # Create the delta
                # delta = -2 * current[k] * Z[:, k]
                new_vals = base_vals - 2 * current[k] * Z[:, k]
                new_norm = np.max(np.abs(new_vals))
                
                if new_norm < best_new_val - 1e-12:
                    best_new_val = new_norm
                    best_flip = k
            
            if best_flip == -1:
                # Local minimum reached
                break
            
            # Apply flip
            current[best_flip] *= -1
            current_val = best_new_val
            
        return current, current_val

    # --- Exhaustive Search Setup ---
    # We will attempt exhaustive search if local search finishes quickly.
    # Total candidates: 2^24 = 16,777,216
    # We fix c[0] = 1 to reduce search space to 2^23 = 8,388,608 (symmetry: P(z) and -P(z) have same norm)
    
    def exhaustive_search():
        if time.time() > deadline - 10.0:
            return None, None
            
        # We need to iterate through all 2^23 combinations of the remaining 23 bits.
        # Batch processing is key.
        
        batch_size = 2048
        total_candidates = 1 << 23 # 23 free bits
        
        best_coeffs = None
        best_val = np.inf
        
        # Precompute Z for a slightly smaller grid for speed in exhaustive?
        # M_exh = 512 is likely enough to distinguish candidates, but might miss the global min if it's close.
        # However, the local search should have found the global min already.
        # Exhaustive is just to be sure or to find an even better one (unlikely).
        # Let's use M_search=2048 for consistency.
        
        # Memory check:
        # Batch: 2048 candidates x 24 coeffs = 49k floats.
        # Values: 2048 x 2048 complex = 4M complex128 = 64MB.
        # This is fine.
        
        for start in range(0, total_candidates, batch_size):
            if time.time() > deadline - 5.0:
                break
                
            end = min(start + batch_size, total_candidates)
            current_batch_size = end - start
            
            # Generate bit patterns for the last 23 coefficients
            # c[0] is fixed to 1
            # We need to map integers [start, end) to 23-bit binary strings
            
            # Using numpy to generate bits is tricky for arbitrary ranges.
            # We can use np.arange(start, end) and bit operations.
            
            indices = np.arange(start, end, dtype=np.uint32)
            
            # Initialize coeffs array: (batch_size, 24)
            coeffs_batch = np.zeros((current_batch_size, N), dtype=np.float64)
            coeffs_batch[:, 0] = 1.0
            
            # Unpack bits for c[1] to c[23]
            # Bit 0 of index corresponds to c[1]
            # Bit 1 corresponds to c[2]
            # ...
            for i in range(23):
                # Extract bit i
                bits = (indices >> i) & 1
                # Map 0 -> -1, 1 -> 1
                coeffs_batch[:, i + 1] = 2.0 * bits - 1.0
                
            # Evaluate sup norm for all candidates in this batch
            # vals = coeffs_batch @ Z.T
            # coeffs_batch shape: (B, 24)
            # Z.T shape: (24, M)
            # vals shape: (B, M)
            
            vals = coeffs_batch @ Z.T
            norms = np.max(np.abs(vals), axis=1)
            
            # Find min in batch
            min_idx = np.argmin(norms)
            min_val = norms[min_idx]
            
            if min_val < best_val:
                best_val = min_val
                best_coeffs = coeffs_batch[min_idx].copy()
                
        return best_coeffs, best_val

    # --- Execution Plan ---
    
    # 1. Run Local Search from Rudin-Shapiro
    rs_coeffs = rudin_shapiro(N)
    best_coeffs, best_val = local_search(rs_coeffs)
    
    # 2. Run Local Search from a few random restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        random_coeffs = rng.choice([-1.0, 1.0], size=N)
        c, v = local_search(random_coeffs, max_iters=500)
        if v < best_val:
            best_val = v
            best_coeffs = c
            
    # 3. If we have time, run Exhaustive Search
    # Local search for N=24 usually finds the global optimum (1.258...)
    # The optimal value is 1.2583758...
    # If our local search found something close, we can verify with exhaustive.
    
    # Check if local search result is likely optimal
    # The optimal sup norm is approx 1.258 * sqrt(24) = 6.15
    # Let's check the value found.
    
    # Exhaustive search is a heavy lift. 
    # 8M candidates.
    # Batch 2048. 4096 batches.
    # Each batch: 2048x24 @ 24x2048 -> 2048x2048 matrix mult.
    # GEMM speed ~ 10 GFLOPS.
    # 2 * 2048 * 24 * 2048 = 201M FLOPs per batch.
    # ~ 20 ms per batch.
    # 4096 batches * 20 ms = 80 seconds.
    # This is too slow for 140s deadline if we want to finish.
    # We need to be smarter or accept that we might not finish exhaustive.
    
    # However, we can reduce M for exhaustive.
    # If M=512, batch time is 4x faster. ~5ms.
    # 4096 * 5ms = 20 seconds. This is feasible!
    # Risk: M=512 might alias the peak.
    # Mitigation: After finding the best candidate from exhaustive (M=512), 
    # refine its score with M=2048 or local search.
    
    def exhaustive_search_fast(M_fast):
        if time.time() > deadline - 10.0:
            return None, None
            
        angles_fast = np.linspace(0, 2 * np.pi, M_fast, endpoint=False)
        k_vec = np.arange(N)
        Z_fast = np.exp(1j * np.outer(angles_fast, k_vec))
        
        batch_size = 4096
        total_candidates = 1 << 23
        
        best_coeffs = None
        best_val = np.inf
        
        for start in range(0, total_candidates, batch_size):
            if time.time() > deadline - 5.0:
                break
                
            end = min(start + batch_size, total_candidates)
            current_batch_size = end - start
            
            indices = np.arange(start, end, dtype=np.uint32)
            coeffs_batch = np.zeros((current_batch_size, N), dtype=np.float64)
            coeffs_batch[:, 0] = 1.0
            
            for i in range(23):
                bits = (indices >> i) & 1
                coeffs_batch[:, i + 1] = 2.0 * bits - 1.0
                
            vals = coeffs_batch @ Z_fast.T
            norms = np.max(np.abs(vals), axis=1)
            
            min_idx = np.argmin(norms)
            min_val = norms[min_idx]
            
            if min_val < best_val:
                best_val = min_val
                best_coeffs = coeffs_batch[min_idx].copy()
                
        return best_coeffs, best_val

    # Try exhaustive search with smaller grid
    # Use M=1024 to be safer
    ex_coeffs, ex_val = exhaustive_search_fast(1024)
    
    if ex_coeffs is not None and ex_val < best_val:
        # Refine the exhaustive result with the high-res local search
        # The exhaustive result might be slightly off due to coarser grid
        refined_coeffs, refined_val = local_search(ex_coeffs, max_iters=100)
        if refined_val < best_val:
            best_val = refined_val
            best_coeffs = refined_coeffs
        elif ex_val < best_val:
            # Fallback to raw exhaustive if local search made it worse (unlikely)
            best_val = ex_val
            best_coeffs = ex_coeffs

    # Convert to int list
    return [int(c) for c in best_coeffs]
