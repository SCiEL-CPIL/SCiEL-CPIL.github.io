<<<<<<< SEARCH
def _local_search(a, b, p, max_iters=5000, time_limit=None):
    """Iterated Local Search with Steepest Descent and Kicks."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    rng = np.random.default_rng()
    
    def _delta_swap(i, j):
        """Compute delta for swapping p[i] and p[j]."""
        pi = p[i]
        pj = p[j]
        
        # Direct terms
        delta_direct = (a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj] + a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]) \
                     - (a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi] + a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj])
        
        # Indirect terms
        col_diff = b[:, pj] - b[:, pi]
        row_diff = b[pi, :] - b[pj, :]
        
        a_col_diff = a[:, i] - a[:, j]
        a_row_diff = a[j, :] - a[i, :]
        
        col_at_p = col_diff[p]
        row_at_p = row_diff[p]
        
        full_col_sum = np.dot(a_col_diff, col_at_p)
        full_row_sum = np.dot(a_row_diff, row_at_p)
        
        # Exclude i and j
        col_excl = full_col_sum - a_col_diff[i] * col_at_p[i] - a_col_diff[j] * col_at_p[j]
        row_excl = full_row_sum - a_row_diff[i] * row_at_p[i] - a_row_diff[j] * row_at_p[j]
        
        return delta_direct + col_excl + row_excl

    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Steepest Descent: find the best swap
        min_delta = 0 # We only accept if < 0
        best_i, best_j = -1, -1
        
        # To speed up, we can sample if n is large, but n=100 is small enough for full scan
        # However, full scan in Python is ~5000 * (2 dots of 100) = 1M ops. 
        # Let's try full scan.
        
        for i in range(n):
            for j in range(i + 1, n):
                d = _delta_swap(i, j)
                if d < min_delta:
                    min_delta = d
                    best_i = i
                    best_j = j
        
        if min_delta < 0:
            # Apply best swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            current_cost += min_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improving swap found, perform a kick
            # Random swap of 2 elements
            i, j = rng.choice(n, size=2, replace=False)
            if i != j:
                d = _delta_swap(i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            # Optional: increase kick strength if stuck for a while? 
            # For simplicity, single swap kick.
            
    return best_p.tolist(), best_cost
=======
def _local_search(a, b, p, max_iters=5000, time_limit=None):
    """Iterated Local Search with vectorized delta computation and adaptive kicks."""
    n = len(a)
    p = np.array(p, dtype=np.int64)
    current_cost = _evaluate(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    rng = np.random.default_rng()
    
    # Precompute some useful arrays
    a_rows = a  # a[i, :] is row i of a
    b_rows = b  # b[j, :] is row j of b
    
    def _compute_deltas_for_i(i, p, a, b):
        """Compute all deltas for swapping p[i] with p[j] for all j > i.
        
        Returns array of deltas for j = i+1, ..., n-1
        """
        pi = p[i]
        
        # For each j, delta = change in cost from swapping p[i] and p[j]
        # We can vectorize this computation
        
        # Direct terms for all j at once
        # a[i,j] * b[pj, pi] - a[i,j] * b[pi, pj] for all j
        pj_all = p[(i+1):]
        
        # b[pj, pi] for all j: this is b[pj_all, pi]
        b_pj_pi = b[pj_all, pi]
        b_pi_pj = b[pi, pj_all]
        
        # a[i, j] for j = i+1, ..., n-1
        a_i_j = a[i, (i+1):]
        
        # a[j, i] for j = i+1, ..., n-1
        a_j_i = a[(i+1):, i]
        
        # a[i, i] and a[j, j]
        a_ii = a[i, i]
        a_jj = a[(i+1):, (i+1):]
        
        # b[pj, pj] and b[pi, pi]
        b_pj_pj = b[pj_all, pj_all]
        b_pi_pi = b[pi, pi]
        
        # Direct part
        delta_direct = a_i_j * (b_pj_pi - b_pi_pj) + a_j_i * (b_pj_pi - b_pi_pj) \
                      + a_ii * (b_pj_pj - b_pi_pi) + a_jj * (b_pi_pi - b_pj_pj)
        
        # Indirect terms: need to compute for each j
        # col_diff[j] = b[:, p[j]] - b[:, pi]
        # row_diff[j] = b[pi, :] - b[p[j], :]
        # 
        # full_col_sum[j] = dot(a[:, i] - a[:, j], b[:, p[j]] - b[:, pi])
        # full_row_sum[j] = dot(a[j, :] - a[i, :], b[pi, :] - b[p[j], :])
        # 
        # This is hard to fully vectorize, but we can do it in a loop over j
        # with vectorized inner operations
        
        a_col_i = a[:, i]
        a_row_i = a[i, :]
        b_col_pi = b[:, pi]
        b_row_pi = b[pi, :]
        
        deltas = np.empty(n - i - 1, dtype=np.float64)
        
        for idx, j in enumerate(range(i + 1, n)):
            pj = p[j]
            
            # Direct part already computed in delta_direct[idx]
            d = delta_direct[idx]
            
            # Indirect column part: sum_k (a[k,i] - a[k,j]) * (b[k,pj] - b[k,pi])
            # = sum_k a[k,i]*b[k,pj] - sum_k a[k,i]*b[k,pi] - sum_k a[k,j]*b[k,pj] + sum_k a[k,j]*b[k,pi]
            # = a_col_i @ b[:,pj] - a_col_i @ b[:,pi] - a[:,j] @ b[:,pj] + a[:,j] @ b[:,pi]
            # = (a_col_i - a[:,j]) @ (b[:,pj] - b[:,pi])
            
            # But we need to exclude i and j
            # Let's compute the full sum and subtract the i and j terms
            
            a_col_diff = a_col_i - a[:, j]
            b_col_diff = b[:, pj] - b_col_pi
            
            full_col = np.dot(a_col_diff, b_col_diff)
            # Subtract i and j terms
            full_col -= a_col_diff[i] * b_col_diff[i]
            full_col -= a_col_diff[j] * b_col_diff[j]
            
            # Indirect row part: sum_k (a[j,k] - a[i,k]) * (b[pi,k] - b[pj,k])
            a_row_diff = a[j, :] - a_row_i
            b_row_diff = b_row_pi - b[pj, :]
            
            full_row = np.dot(a_row_diff, b_row_diff)
            # Subtract i and j terms
            full_row -= a_row_diff[i] * b_row_diff[i]
            full_row -= a_row_diff[j] * b_row_diff[j]
            
            deltas[idx] = d + full_col + full_row
        
        return deltas

    stuck_count = 0
    kick_strength = 1
    
    for iteration in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Find best swap using vectorized delta computation
        min_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n - 1):
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break
            deltas = _compute_deltas_for_i(i, p, a, b)
            min_idx = np.argmin(deltas)
            min_d = deltas[min_idx]
            if min_d < min_delta:
                min_delta = min_d
                best_i = i
                best_j = i + 1 + min_idx
        
        if min_delta < 0:
            # Apply best swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            current_cost += min_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            stuck_count = 0
            kick_strength = 1
        else:
            # No improving swap found, perform a kick
            stuck_count += 1
            # Increase kick strength periodically
            if stuck_count % 10 == 0:
                kick_strength = min(kick_strength + 1, 4)
            
            # Perform kick_strength random swaps
            for _ in range(kick_strength):
                i, j = rng.choice(n, size=2, replace=False)
                if i != j:
                    # Compute delta for this specific swap
                    pi, pj = p[i], p[j]
                    
                    delta_direct = (a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj] + a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]) \
                                 - (a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi] + a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj])
                    
                    a_col_diff = a[:, i] - a[:, j]
                    b_col_diff = b[:, pj] - b[:, pi]
                    full_col = np.dot(a_col_diff, b_col_diff)
                    full_col -= a_col_diff[i] * b_col_diff[i]
                    full_col -= a_col_diff[j] * b_col_diff[j]
                    
                    a_row_diff = a[j, :] - a[i, :]
                    b_row_diff = b[pi, :] - b[pj, :]
                    full_row = np.dot(a_row_diff, b_row_diff)
                    full_row -= a_row_diff[i] * b_row_diff[i]
                    full_row -= a_row_diff[j] * b_row_diff[j]
                    
                    d = delta_direct + full_col + full_row
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist(), best_cost
>>>>>>> REPLACE

