   def dual_greedy_construct(randomize=True, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        cover_counts = [0] * num_t_subsets
        
        while uncovered:
            if time.time() > deadline:
                return None, False
            
            max_unique = 0
            candidates_with_max = []
            
            # Pass 1: Find max unique coverage
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                unique = popcount(uncovered & masks[i])
                if unique > max_unique:
                    max_unique = unique
                    candidates_with_max = [i]
                elif unique == max_unique:
                    candidates_with_max.append(i)
            
            if max_unique == 0:
                return current_blocks, False
            
            # Pass 2: Among max unique candidates, minimize overlap
            best_score = float('inf') # We want min overlap
            best_candidate = None
            tie_candidates = []
            
            for i in candidates_with_max:
                overlap_sum = 0
                m = masks[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    overlap_sum += cover_counts[j]
                    m ^= lsb
                
                if overlap_sum < best_score:
                    best_score = overlap_sum
                    best_candidate = i
                    tie_candidates = [i]
                elif overlap_sum == best_score:
                    tie_candidates.append(i)
            
            if randomize and len(tie_candidates) > 1:
                chosen_idx = random.choice(tie_candidates)
            else:
                chosen_idx = tie_candidates[0]
            
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update states
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            m = masks[chosen_idx]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        return current_blocks, True
   