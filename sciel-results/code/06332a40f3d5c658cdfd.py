def construct():
    import random
    import time

    def count_nonzero_square_fast(terms):
        """
        terms: list of (index, coeff) pairs, sorted by index.
        Returns count of nonzero coefficients in the square.
        """
        square_coeffs = {}
        n = len(terms)
        for i in range(n):
            idx_i, c_i = terms[i]
            for j in range(i, n):
                idx_j, c_j = terms[j]
                prod = c_i * c_j
                sum_idx = idx_i + idx_j
                if sum_idx in square_coeffs:
                    square_coeffs[sum_idx] += prod
                else:
                    square_coeffs[sum_idx] = prod
                
                if i != j:
                    # The term c_j * c_i at index idx_j + idx_i is the same as above
                    # Wait, the square is sum_{i,j} c_i c_j x^{i+j}.
                    # My loop i<=j handles the diagonal and off-diagonals.
                    # If i != j, the term (i,j) and (j,i) both contribute to the same coefficient.
                    # So I should add 2 * prod?
                    # Let's re-verify.
                    # P^2 = (sum c_i x^i)^2 = sum_i sum_j c_i c_j x^{i+j}.
                    # Coeff of x^k is sum_{i+j=k} c_i c_j.
                    # If I iterate i from 0 to n-1 and j from 0 to n-1, I count each pair.
                    # If I iterate i <= j, I count (i,i) once and (i,j) once.
                    # But (i,j) and (j,i) are distinct terms in the expansion if i!=j.
                    # So coeff is c_i c_i + 2 * c_i c_j (if i!=j).
                    # My current code adds prod once for (i,j).
                    # I need to add prod again if i != j?
                    # Yes.
                    square_coeffs[sum_idx] += prod
        
        return sum(1 for v in square_coeffs.values() if v != 0)

    def terms_to_list(terms):
        """Convert list of (idx, coeff) to coefficient list in ascending degree."""
        if not terms:
            return []
        max_idx = max(idx for idx, _ in terms)
        if max_idx >= 41:
            return None # Invalid
        coeff_list = [0] * (max_idx + 1)
        for idx, c in terms:
            coeff_list[idx] = c
        # Ensure length <= 41 and nonzero final coeff
        if len(coeff_list) > 41:
            return None
        if coeff_list[-1] == 0:
            # This shouldn't happen if max_idx is from a nonzero term
            pass
        return coeff_list

    def valid_candidate_list(lst):
        if lst is None:
            return False
        if len(lst) > 41:
            return False
        if not lst or lst[-1] == 0:
            return False
        if sum(1 for c in lst if c != 0) != 12:
            return False
        if any(abs(c) > 64 for c in lst):
            return False
        return True

    def random_terms():
        """Generate a random valid set of 12 terms."""
        indices = random.sample(range(41), 12)
        indices.sort()
        terms = []
        for idx in indices:
            # Coeffs in [-64, 64], nonzero
            c = random.choice([x for x in range(-64, 65) if x != 0])
            # To keep numbers small for better cancellation potential, bias towards small ints
            # But let's stick to small range initially
            c = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5])
            terms.append((idx, c))
        return terms

    def hill_climb(terms, time_limit):
        """
        Perform local search on the terms representation.
        Returns best terms found.
        """
        current_terms = [t[:] for t in terms]
        current_score = count_nonzero_square_fast(current_terms)
        best_terms = [t[:] for t in current_terms]
        best_score = current_score
        
        start_time = time.time()
        no_improvement_count = 0
        max_no_improvement = 2000 # Reset after this many failures? Or just continue.
        
        while time.time() - start_time < time_limit:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Move index of one term
                term_idx = random.randint(0, 11)
                old_idx, old_c = current_terms[term_idx]
                new_idx = random.randint(0, 40)
                
                # Ensure new_idx is not occupied by another term
                occupied = set(idx for idx, _ in current_terms)
                if new_idx in occupied and new_idx != old_idx:
                    continue
                if new_idx == old_idx:
                    continue
                    
                current_terms[term_idx] = (new_idx, old_c)
                # Keep sorted for potential optimizations, though not strictly needed for count
                current_terms.sort(key=lambda x: x[0])
                
                new_score = count_nonzero_square_fast(current_terms)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_terms = [t[:] for t in current_terms]
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # Revert
                    current_terms[term_idx] = (old_idx, old_c)
                    current_terms.sort(key=lambda x: x[0])
                    no_improvement_count += 1
                    
            elif move_type < 0.7:
                # Flip sign of one term
                term_idx = random.randint(0, 11)
                old_idx, old_c = current_terms[term_idx]
                new_c = -old_c
                current_terms[term_idx] = (old_idx, new_c)
                
                new_score = count_nonzero_square_fast(current_terms)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_terms = [t[:] for t in current_terms]
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    current_terms[term_idx] = (old_idx, old_c)
                    no_improvement_count += 1
                    
            elif move_type < 0.9:
                # Change magnitude of one term
                term_idx = random.randint(0, 11)
                old_idx, old_c = current_terms[term_idx]
                # Pick a new small magnitude
                new_c = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6])
                if new_c == old_c:
                    continue
                current_terms[term_idx] = (old_idx, new_c)
                
                new_score = count_nonzero_square_fast(current_terms)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_terms = [t[:] for t in current_terms]
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    current_terms[term_idx] = (old_idx, old_c)
                    no_improvement_count += 1
                    
            else:
                # Swap indices of two terms (effectively moving both)
                i1, i2 = random.sample(range(12), 2)
                idx1, c1 = current_terms[i1]
                idx2, c2 = current_terms[i2]
                current_terms[i1] = (idx2, c1)
                current_terms[i2] = (idx1, c2)
                current_terms.sort(key=lambda x: x[0])
                
                new_score = count_nonzero_square_fast(current_terms)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_terms = [t[:] for t in current_terms]
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # Revert
                    current_terms[i1] = (idx1, c1)
                    current_terms[i2] = (idx2, c2)
                    current_terms.sort(key=lambda x: x[0])
                    no_improvement_count += 1

            # If stuck for too long, do a random jump to escape local minimum
            if no_improvement_count > 500:
                # Perturb 2 random terms
                for _ in range(2):
                    term_idx = random.randint(0, 11)
                    idx, c = current_terms[term_idx]
                    new_idx = random.randint(0, 40)
                    occupied = set(i for i, _ in current_terms)
                    if new_idx not in occupied or new_idx == idx:
                         current_terms[term_idx] = (new_idx, c)
                current_terms.sort(key=lambda x: x[0])
                current_score = count_nonzero_square_fast(current_terms)
                no_improvement_count = 0

        return best_terms, best_score

    # Main loop: Random Restarts
    global_start = time.time()
    total_time = 55.0
    best_global_score = float('inf')
    best_global_terms = random_terms()
    
    # Initial evaluation
    best_global_score = count_nonzero_square_fast(best_global_terms)
    
    while time.time() - global_start < total_time:
        # Determine time for this restart
        time_left = total_time - (time.time() - global_start)
        if time_left < 0.1:
            break
        
        # Spend a fraction of remaining time on this restart
        # Or fixed time per restart?
        # Let's do short restarts to explore more
        restart_time = min(1.0, time_left / 2) 
        
        initial_terms = random_terms()
        best_terms, best_score = hill_climb(initial_terms, restart_time)
        
        if best_score < best_global_score:
            best_global_score = best_score
            best_global_terms = best_terms
            # If we get a very good score, we can try to polish it further
            # But hill_climb already does that.
            
    # Convert best terms to list
    result_list = terms_to_list(best_global_terms)
    
    # Fallback if something went wrong
    if not valid_candidate_list(result_list):
        # Construct a valid fallback
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        coeffs = [1]*12
        result_list = coeffs
        
    return result_list
