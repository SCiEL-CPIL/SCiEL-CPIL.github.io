import time
import random
import numpy as np

def construct():
    # Deadline: 55 seconds to be safe
    deadline = time.time() + 55.0
    
    # Target sums: 128 to 256 inclusive
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    ALL_ELEMENTS = list(range(0, 129))
    ALL_ELEMENTS_SET = set(ALL_ELEMENTS)
    
    # Precompute for each element x, the list of (sum_index, count) it contributes to
    # When x is in A, for each a in A, x contributes to sum a+x
    # We'll compute this on the fly but cache the structure
    
    # Helper to calculate multiplicities for a given set A using FFT convolution
    def calc_mult(A):
        """Compute representation multiplicities using FFT convolution.
        For unordered pairs with equal summands allowed:
        M[s] = (num_ordered_pairs(s) + num_equal_pairs(s)) // 2
        where num_ordered_pairs counts (a,b) with a+b=s, a!=b
        and num_equal_pairs counts (a,a) with 2a=s.
        """
        # Create binary indicator array of length 257 (0..256)
        indicator = np.zeros(257, dtype=np.float64)
        for a in A:
            indicator[a] = 1.0
        # FFT convolution: indicator * indicator gives ordered pair counts
        fft_size = 1  # next power of 2 >= 513
        while fft_size < 513:
            fft_size *= 2
        fft_a = np.fft.fft(indicator, n=fft_size)
        conv = np.fft.ifft(fft_a * fft_a).real
        # conv[s] = number of ordered pairs (a,b) with a+b=s, a,b in A
        # For unordered with equal: M[s] = (conv[s] + eq[s]) / 2
        # where eq[s] = 1 if s is even and s/2 in A, else 0
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        for s_idx in range(NUM_SUMS):
            s = MIN_SUM + s_idx
            conv_val = int(round(conv[s]))
            # Check if s/2 is in A and s is even
            eq_val = 0
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= 128 and half in A:
                    eq_val = 1
            M[s_idx] = (conv_val + eq_val) // 2
        return M
    
    # Helper: compute contribution of element x to each sum
    # Returns an array of length NUM_SUMS with the counts x contributes
    def get_contribution(x, A):
        """Get how much x contributes to each sum when A is the current set."""
        contrib = np.zeros(NUM_SUMS, dtype=np.int32)
        for a in A:
            s = a + x
            if MIN_SUM <= s <= MAX_SUM:
                contrib[s - MIN_SUM] += 1
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            contrib[s - MIN_SUM] += 1
        return contrib
    
    # Helper: remove element x from A and update M incrementally (vectorized)
    def remove_from_A(x, A, M):
        """Remove x from A, update M. Returns (new_A, new_M) or (None, None) if invalid."""
        # x contributes to sums x+a for all a in A (a != x), and 2*x once
        # Vectorized: compute all sums x + A
        A_arr = np.array(sorted(A), dtype=np.int32)
        sums = x + A_arr  # includes x+x
        # For unordered: x contributes 1 to each sum x+a where a in A, a != x
        # and 1 to 2x (the equal pair)
        # So total contribution: for each a in A, a != x: sum x+a gets 1
        # and sum 2x gets 1
        # But since we're iterating over all a in A including x, 
        # x+x appears once in sums, and x+a for a!=x appear once each
        # In the unordered count, (x,a) for a!=x is counted once, (x,x) once
        # So contribution is: 1 for each a in A (a can equal x, giving 2x)
        # Wait: in calc_mult, for unordered pairs, (a,b) with a<=b.
        # When we remove x, we lose:
        #   (x, a) for all a in A with a >= x (since unordered, we count a<=b)
        #   Actually no: the representation count M[s] counts all unordered pairs.
        #   Removing x removes all pairs involving x: (x, a) for all a in A.
        #   Each such pair is unordered, so (x,a) = (a,x).
        #   The number of pairs involving x is |A| (one for each a in A, including a=x).
        #   But wait: (x,x) is one pair, (x,a) for a!=x is one pair each.
        #   So total pairs removed = |A|.
        #   The sums affected: x+a for each a in A.
        
        # Vectorized update
        new_M = M.copy()
        valid = (sums >= MIN_SUM) & (sums <= MAX_SUM)
        if np.any(valid):
            indices = sums[valid] - MIN_SUM
            np.add.at(new_M, indices, -1)
        if np.min(new_M) < 1:
            return None, None
        new_A = A - {x}
        return new_A, new_M
    
    # Helper: add element x to A and update M incrementally (vectorized)
    def add_to_A(x, A, M):
        """Add x to A, update M. Returns (new_A, new_M)."""
        # Adding x adds pairs (x, a) for all a in A, plus (x, x)
        # That's |A| + 1 pairs total
        A_arr = np.array(sorted(A), dtype=np.int32)
        sums = x + A_arr  # x+a for each a in A
        # Also add 2*x for the (x,x) pair
        all_sums = np.concatenate([sums, [2*x]])
        new_M = M.copy()
        valid = (all_sums >= MIN_SUM) & (all_sums <= MAX_SUM)
        if np.any(valid):
            indices = all_sums[valid] - MIN_SUM
            np.add.at(new_M, indices, 1)
        new_A = A | {x}
        return new_A, new_M
    
    # Smoother scoring: use soft-max instead of hard max
    def soft_max_score(M, beta=2.0):
        """Compute a smooth approximation to max(M)."""
        # Avoid overflow
        M_float = M.astype(np.float64)
        max_val = np.max(M_float)
        if max_val == 0:
            return 0.0
        exps = np.exp(beta * (M_float - max_val))
        return max_val + np.log(np.sum(exps)) / beta
    
    # Better initial sets: more diverse
    initial_sets = []
    # Structured sets
    initial_sets.append(set(range(64, 129)))
    initial_sets.append(set(range(32, 97)))
    initial_sets.append(set(range(0, 129, 2)))
    initial_sets.append(set(range(64, 129)) | set(range(0, 32, 2)))
    initial_sets.append(set(range(48, 113)))
    initial_sets.append(set(range(0, 65)) | set(range(64, 129, 2)))
    # Random sets with good coverage
    for _ in range(3):
        # Try to build a set with good coverage
        A_rand = set()
        # Start with some spread
        A_rand.update(random.sample(ALL_ELEMENTS, 40))
        # Add more to ensure coverage
        for _ in range(50):
            if len(A_rand) >= 50:
                break
            A_rand.add(random.choice(ALL_ELEMENTS))
        initial_sets.append(A_rand)
    
    best_A = None
    best_score = float('inf')
    best_A_set = None
    
    for init_set in initial_sets:
        M_init = calc_mult(init_set)
        if np.min(M_init) < 1:
            continue  # Skip invalid initial sets
        
        A = set(init_set)
        M = M_init
        current_score = int(np.max(M))
        
        if current_score < best_score:
            best_score = current_score
            best_A = set(A)
            best_A_set = set(A)
    
    if best_A is None:
        # Fallback to original
        A = set(range(64, 129))
        M = calc_mult(A)
        best_A = set(A)
        best_A_set = set(A)
        best_score = int(np.max(M))
    
    # Simulated Annealing with improved moves
    T = 5.0
    cooling = 0.998  # Slightly faster cooling to reach low T sooner for exploitation
    min_T = 0.01
    
    # Precompute contribution vectors for all elements (will be updated as A changes)
    # For now, compute on the fly
    
    while time.time() < deadline:
        # Choose a move type with bias toward removal/swap
        move_type = random.random()
        
        if move_type < 0.45:
            # Remove an element
            if len(A) <= 15:  # Don't make set too small
                T = max(T * cooling, min_T)
                continue
            
            # Prefer removing elements that contribute to high-multiplicity sums
            # Score each element by how much it contributes to the max
            A_list = list(A)
            if len(A_list) <= 20:
                # Small set: evaluate all
                scores = []
                for x in A_list:
                    contrib = get_contribution(x, A)
                    # How much does removing x reduce the max?
                    # Approximate: max of (M - contrib)
                    temp_M = M - contrib
                    if np.min(temp_M) >= 1:
                        scores.append((int(np.max(temp_M)), x))
                    else:
                        scores.append((1000, x))  # Invalid
                # Sort by resulting max (we want to remove elements that keep max low)
                scores.sort(key=lambda t: t[0])
                # Pick from top few
                k = min(3, len(scores))
                _, x = scores[random.randint(0, k - 1)]
            else:
                # Large set: random but biased
                # Compute contribution magnitude for a sample
                sample_size = min(20, len(A_list))
                sample = random.sample(A_list, sample_size)
                contrib_mags = []
                for x in sample:
                    contrib = get_contribution(x, A)
                    # Magnitude of contribution to high sums
                    high_sums = M > (np.max(M) - 1)
                    contrib_mag = np.sum(contrib * high_sums)
                    contrib_mags.append((contrib_mag, x))
                contrib_mags.sort(reverse=True)
                # Pick from elements with high contribution to peak sums
                if contrib_mags:
                    top_k = min(5, len(contrib_mags))
                    _, x = contrib_mags[random.randint(0, top_k - 1)]
                else:
                    x = random.choice(A_list)
            
            new_A, new_M = remove_from_A(x, A, M)
            if new_M is not None:
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    A = new_A
                    M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
                        best_A_set = set(A)
        
        elif move_type < 0.85:
            # Swap: remove x, add y
            if len(A) == 0:
                T = max(T * cooling, min_T)
                continue
            
            # Choose x to remove (biased toward high contributors)
            A_list = list(A)
            if len(A_list) <= 20:
                scores = []
                for x in A_list:
                    temp_A, temp_M = remove_from_A(x, A, M)
                    if temp_M is not None:
                        scores.append((int(np.max(temp_M)), x))
                    else:
                        scores.append((1000, x))
                scores.sort(key=lambda t: t[0])
                k = min(3, len(scores))
                _, x = scores[random.randint(0, k - 1)]
            else:
                x = random.choice(A_list)
            
            # Choose y to add (biased toward filling gaps)
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T = max(T * cooling, min_T)
                continue
            
            # Bias toward elements that cover low-multiplicity sums
            # Find sums with low multiplicity
            low_sums = M <= (np.max(M) + np.min(M)) // 2
            low_sum_indices = np.where(low_sums)[0]
            
            if len(low_sum_indices) > 0 and len(candidates) <= 50:
                # Score candidates by how well they cover low sums
                cand_scores = []
                for y in candidates:
                    # What sums does y create?
                    good_sums = 0
                    for a in A:
                        s = a + y
                        if MIN_SUM <= s <= MAX_SUM:
                            idx = s - MIN_SUM
                            if M[idx] <= (np.max(M) + np.min(M)) // 2:
                                good_sums += 1
                    s = 2 * y
                    if MIN_SUM <= s <= MAX_SUM:
                        idx = s - MIN_SUM
                        if M[idx] <= (np.max(M) + np.min(M)) // 2:
                            good_sums += 1
                    cand_scores.append((good_sums, y))
                cand_scores.sort(reverse=True)
                top_k = min(5, len(cand_scores))
                _, y = cand_scores[random.randint(0, top_k - 1)]
            else:
                y = random.choice(candidates)
            
            # Simulate: remove x then add y
            temp_A, temp_M = remove_from_A(x, A, M)
            if temp_M is not None:
                new_A, new_M = add_to_A(y, temp_A, temp_M)
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    A = new_A
                    M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
                        best_A_set = set(A)
        
        else:
            # Add an element (less common)
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T = max(T * cooling, min_T)
                continue
            
            # Bias toward elements that cover low-multiplicity sums
            low_threshold = (np.max(M) + np.min(M)) // 2
            if len(candidates) <= 50:
                cand_scores = []
                for y in candidates:
                    good_sums = 0
                    for a in A:
                        s = a + y
                        if MIN_SUM <= s <= MAX_SUM:
                            idx = s - MIN_SUM
                            if M[idx] <= low_threshold:
                                good_sums += 1
                    s = 2 * y
                    if MIN_SUM <= s <= MAX_SUM:
                        idx = s - MIN_SUM
                        if M[idx] <= low_threshold:
                            good_sums += 1
                    cand_scores.append((good_sums, y))
                cand_scores.sort(reverse=True)
                top_k = min(5, len(cand_scores))
                _, y = cand_scores[random.randint(0, top_k - 1)]
            else:
                y = random.choice(candidates)
            
            new_A, new_M = add_to_A(y, A, M)
            score_new = int(np.max(new_M))
            delta = score_new - current_score
            # Adding usually increases max, so only accept with low probability
            if delta <= 0 or (T > min_T and random.random() < np.exp(-delta / T) * 0.3):
                A = new_A
                M = new_M
                current_score = score_new
        
        # Cool down
        T = max(T * cooling, min_T)
    
    # Final polish: vectorized aggressive removal pass
    # Try removing each element and check if max stays <= best_score
    removed = True
    polish_deadline = time.time() + min(15, max(0, deadline - time.time()))
    while removed and time.time() < polish_deadline:
        removed = False
        A_list = list(A)
        random.shuffle(A_list)
        for x in A_list:
            if x not in A:
                continue
            new_A, new_M = remove_from_A(x, A, M)
            if new_M is not None:
                score_new = int(np.max(new_M))
                if score_new <= best_score:
                    A = new_A
                    M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
                        best_A_set = set(A)
                    removed = True
    
    # Second polish: try swaps that maintain or improve score
    if time.time() < deadline - 2:
        swap_deadline = time.time() + min(8, max(0, deadline - time.time() - 2))
        improved = True
        while improved and time.time() < swap_deadline:
            improved = False
            A_list = list(A)
            random.shuffle(A_list)
            for x in A_list:
                if x not in A:
                    continue
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                random.shuffle(candidates)
                # Try up to 30 candidates for speed
                for y in candidates[:30]:
                    temp_A, temp_M = remove_from_A(x, A, M)
                    if temp_M is None:
                        continue
                    new_A, new_M = add_to_A(y, temp_A, temp_M)
                    score_new = int(np.max(new_M))
                    if score_new < best_score:
                        A = new_A
                        M = new_M
                        current_score = score_new
                        best_score = score_new
                        best_A = set(A)
                        best_A_set = set(A)
                        improved = True
                        break
    
    # Third polish: try adding elements that allow further removals
    if time.time() < deadline - 1:
        add_deadline = time.time() + min(3, max(0, deadline - time.time() - 1))
        candidates = [y for y in ALL_ELEMENTS if y not in A]
        random.shuffle(candidates)
        for y in candidates[:20]:
            new_A, new_M = add_to_A(y, A, M)
            # Now try to remove elements from new_A to see if we can get lower max
            best_temp = None
            best_temp_score = int(np.max(new_M))
            for x in list(new_A):
                if x == y:
                    continue
                temp_A2, temp_M2 = remove_from_A(x, new_A, new_M)
                if temp_M2 is not None:
                    s2 = int(np.max(temp_M2))
                    if s2 < best_temp_score:
                        best_temp_score = s2
                        best_temp = (temp_A2, temp_M2)
            if best_temp is not None and best_temp_score < best_score:
                A, M = best_temp
                current_score = best_temp_score
                best_score = best_temp_score
                best_A = set(A)
                best_A_set = set(A)
    
    return list(best_A)

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# Note: The task asks to keep the function names and signatures.
# I will place the code in the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
