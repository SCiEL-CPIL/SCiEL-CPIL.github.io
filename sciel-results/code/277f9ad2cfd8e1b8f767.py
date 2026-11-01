<<<<<<< SEARCH
def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors (balls of radius 1)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # balls[w] contains w and all words at distance 1 from w
    # There are 1 + 2*n = 17 such words
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    all_indices = np.arange(total, dtype=np.int32)
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Greedy Loop
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break # Time out for this iteration
                
            # Count coverage for each candidate among uncovered words
            # For each word c, count how many uncovered words are in balls[c]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen if uncovered_mask has True
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search
        # Phase 1: Remove redundant
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable: min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Swap / Replace
        # Try to replace a codeword with another to allow further reduction
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                # First, try to remove (might have become removable due to previous swaps in this pass)
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Try to swap c with some c'
                swapped = False
                # Try a few random candidates
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    c_prime = all_indices[np.random.randint(total)]
                    if c_prime == c:
                        continue
                    
                    # Check if replacing c with c_prime is valid
                    # Temporarily remove c
                    cover_count[balls[c]] -= 1
                    
                    # Identify critical words for c (words covered only by c)
                    critical_in_c = (cover_count[balls[c]] == 1)
                    
                    if np.any(critical_in_c):
                        critical_indices = balls[c][critical_in_c]
                        # Check if all critical_indices are in balls[c_prime]
                        c_prime_set = set(balls[c_prime].tolist())
                        valid = True
                        for idx in critical_indices:
                            if idx not in c_prime_set:
                                valid = False
                                break
                        if valid:
                            # Swap is valid
                            cover_count[balls[c_prime]] += 1
                            new_selected.append(c_prime)
                            swapped = True
                            improved = True
                            break
                    else:
                        # No critical words, so c was removable, but we checked that above.
                        # This case shouldn't happen if the removal check was robust.
                        # But if it does, the swap is trivially valid?
                        # If no critical words, removing c is safe.
                        # So we should have removed it.
                        pass

                    # If not swapped, revert removal of c
                    if not swapped:
                        cover_count[balls[c]] += 1
                
                if not swapped:
                    # Keep c
                    new_selected.append(c)
            
            selected = new_selected
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                # We hit the known optimum.
                # Break early to save time.
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
=======
def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors (balls of radius 1)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # balls[w] contains w and all words at distance 1 from w
    # There are 1 + 2*n = 17 such words
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    all_indices = np.arange(total, dtype=np.int32)
    
    best_size = total
    best_set = None
    
    # Solve LP relaxation to get dual variables
    # Primal: min sum(x_c) s.t. for all w: sum_{c in ball(w)} x_c >= 1, x_c >= 0
    # This is a large LP (6561 vars, 6561 constraints), but we can use scipy
    # Actually, let's think about this differently. The LP is:
    # min 1^T x
    # s.t. A x >= 1, x >= 0
    # where A[w,c] = 1 if c is in ball(w)
    # 
    # We can solve the dual instead:
    # max 1^T y
    # s.t. A^T y <= 1, y >= 0
    # 
    # The dual has 6561 variables and 6561 constraints too.
    # Let's try solving the primal with linprog.
    
    # Build constraint matrix A where A[w,c] = 1 if c covers w
    # A is total x total, sparse-ish (each row has 17 ones)
    # For linprog, we need Ax <= b form, so: -A x <= -1
    
    # This is a 6561x6561 LP which might be too slow. Let me try a different approach.
    # Instead, let's use the dual variables from a simpler heuristic.
    
    # Alternative: Use greedy with a smarter scoring based on LP-like weighting
    # Or: Solve a smaller LP by exploiting symmetry
    
    # Actually, let me try the LP but with a time limit
    # The LP might be solvable in reasonable time with scipy's high-level solver
    
    # Let me try: solve the dual LP which might be more natural
    # Dual: max sum(y_w) s.t. for each c: sum_{w in ball(c)} y_w <= 1, y_w >= 0
    
    # This is still 6561 vars/constraints. Let me just try it.
    
    # c_dual = -ones(total)  (we want to maximize, linprog minimizes)
    # A_dual: for each codeword c, constraint sum_{w in ball(c)} y_w <= 1
    # So A_dual[c, w] = 1 if w is in ball(c), i.e., A_dual = A.T where A[w,c]=1 if c in ball(w)
    # b_dual = ones(total)
    
    # Actually, let me build it properly:
    # A_matrix[w, c] = 1 if c is in the ball around w
    # For the dual, we need A_dual[c, w] = 1 if w is in ball(c)
    # But ball(c) is symmetric: c in ball(w) iff w in ball(c)
    # So A_dual = A_matrix.T
    
    # Let me build A_matrix first
    A_matrix = np.zeros((total, total), dtype=np.float32)
    for w in range(total):
        for c in balls[w]:
            A_matrix[w, c] = 1.0
    
    # Solve dual LP: max 1^T y s.t. A^T y <= 1, y >= 0
    # In linprog form: min -1^T y s.t. A^T y <= 1, y >= 0
    c_obj = -np.ones(total, dtype=np.float32)
    A_ub = A_matrix.T  # shape (total, total)
    b_ub = np.ones(total, dtype=np.float32)
    
    # Set time limit for LP
    lp_start = time.time()
    lp_result = None
    try:
        lp_result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * total, 
                           method='highs', options={'time_limit': 30})
    except Exception:
        pass
    
    lp_elapsed = time.time() - lp_start
    
    # If LP succeeded, use dual variables to guide greedy
    if lp_result is not None and lp_result.success:
        # lp_result.y is the dual of the primal, which corresponds to our y variables
        # Actually, for min c^T x s.t. Ax <= b, x >= 0, the dual is max b^T y s.t. A^T y <= c, y <= 0
        # Our dual was max 1^T y s.t. A^T y <= 1, y >= 0
        # So we solved: min -1^T y s.t. A^T y <= 1, y >= 0
        # The dual of THIS problem would give us the primal variables
        # But we want y itself, which is lp_result.x
        y_weights = lp_result.x  # shape (total,)
        
        # Use these weights to guide a greedy algorithm
        # Greedy: at each step, pick c that maximizes sum of y_weights over uncovered words in ball(c)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask) and time.time() < deadline - 15.0:
            # Score each candidate: sum of y_weights for uncovered words it covers
            scores = (uncovered_mask[balls] * y_weights[balls]).sum(axis=1)
            
            if np.max(scores) <= 0:
                # Fallback to count-based
                scores = uncovered_mask[balls].sum(axis=1)
            
            best_idx = np.argmax(scores)
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        if np.any(uncovered_mask):
            # Fallback: add missing words
            for w in range(total):
                if uncovered_mask[w]:
                    selected.append(w)
                    cover_count[balls[w]] += 1
                    uncovered_mask = (cover_count == 0)
        
        # Local search: remove redundant
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Swap phase
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                swapped = False
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    c_prime = all_indices[np.random.randint(total)]
                    if c_prime == c:
                        continue
                    
                    cover_count[balls[c]] -= 1
                    critical_in_c = (cover_count[balls[c]] == 1)
                    
                    if np.any(critical_in_c):
                        critical_indices = balls[c][critical_in_c]
                        c_prime_set = set(balls[c_prime].tolist())
                        valid = all(idx in c_prime_set for idx in critical_indices)
                        if valid:
                            cover_count[balls[c_prime]] += 1
                            new_selected.append(c_prime)
                            swapped = True
                            improved = True
                            break
                    else:
                        pass
                    if not swapped:
                        cover_count[balls[c]] += 1
                
                if not swapped:
                    new_selected.append(c)
            
            selected = new_selected
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 486:
                break
    
    # Run additional randomized greedy iterations if time permits
    while time.time() < deadline - 5.0:
        # Randomized greedy with LP-inspired weighting if available
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask) and time.time() < deadline - 10.0:
            counts = uncovered_mask[balls].sum(axis=1)
            if np.max(counts) == 0:
                break
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        if np.any(uncovered_mask):
            continue
        
        # Remove redundant
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Swap
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                swapped = False
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    c_prime = all_indices[np.random.randint(total)]
                    if c_prime == c:
                        continue
                    cover_count[balls[c]] -= 1
                    critical_in_c = (cover_count[balls[c]] == 1)
                    if np.any(critical_in_c):
                        critical_indices = balls[c][critical_in_c]
                        c_prime_set = set(balls[c_prime].tolist())
                        valid = all(idx in c_prime_set for idx in critical_indices)
                        if valid:
                            cover_count[balls[c_prime]] += 1
                            new_selected.append(c_prime)
                            swapped = True
                            improved = True
                            break
                    else:
                        pass
                    if not swapped:
                        cover_count[balls[c]] += 1
                if not swapped:
                    new_selected.append(c)
            selected = new_selected
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 486:
                break
    
    # Convert best_set to output format
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
>>>>>>> REPLACE

