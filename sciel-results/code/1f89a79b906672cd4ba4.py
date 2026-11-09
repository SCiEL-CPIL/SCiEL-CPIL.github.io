<<<<<<< SEARCH
    # 2. Helper Functions
    
    def greedy_construct(ordering):
        """Greedy select in given order using bitmask coprimality."""
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask

    def fast_hill_climb(chosen, chosen_set, chosen_mask, current_score, max_rounds=100):
        """
        Best-improvement local search with 1-add and 1-swap moves.
        Optimized for speed using bitmasks.
        """
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_gain = 0.0
            best_move = None  # ('add', b) or ('swap', a, b)
            
            # Try 1-add: add one element if possible
            # We iterate in descending score order to find the best add quickly
            # But since we want the GLOBAL best move, we must check all valid ones
            # However, scores are sorted by a (larger a -> smaller denominator -> larger score).
            # So iterating 511 down to 1 checks highest potential gains first.
            # We can break early if we find a gain that is maximal possible? No, because 
            # swapping might yield a larger net gain than adding.
            # But adding a high score item is always better than adding a low score item.
            
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                if factor_masks[b] & chosen_mask:
                    continue
                gain = scores[b]
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_move = ('add', b)
                    # Optimization: Since scores are descending, the first valid 'add' 
                    # is the best possible 'add'. We still need to check swaps though.
                    # We can't break here because a swap might be better.
            
            # Try 1-swap: remove one, add one
            # To optimize, we only consider removing 'a' if it has a low score, 
            # or if the best available 'b' is significantly higher.
            # A simple optimization: Only try swapping 'a' if scores[b_max] > scores[a].
            # Since we iterate b in descending order, the first valid b for a given a 
            # is the best b for that a.
            
            # To make this faster, we can limit which 'a's we try to remove.
            # If scores[a] is very high, it's unlikely swapping it for anything else improves score.
            # Specifically, if scores[a] >= max_possible_score_of_b, no swap involving removing a is useful.
            # max_possible_score is scores[511].
            
            # Let's iterate 'a' in ascending order of score (i.e., small 'a' first, which have low scores? No, small 'a' have LARGE scores? No.
            # score = 1/(512-a). 
            # a=1 -> 1/511 ~ 0.0019
            # a=511 -> 1/1 = 1.0
            # So large 'a' have high scores. We want to KEEP high scores.
            # We want to REMOVE low scores (small 'a').
            # So we should iterate 'a' from 1 upwards (lowest scores first) for removal candidates.
            
            for a in range(1, 512):
                if a not in chosen_set:
                    continue
                
                # If the best possible score (1.0) is not greater than current score of a, 
                # and we already have a good add move, maybe skip? 
                # Actually, if scores[a] is 1.0, we can't swap it for anything better.
                # If scores[a] is 0.5, we can swap for 1.0 (gain 0.5).
                
                temp_mask = chosen_mask & ~factor_masks[a]
                
                # Find best b to add if we remove a
                # Iterate b in descending score order
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_move = ('swap', a, b)
                        # We found a valid b for this a. Since b is descending, this is the best b for this a.
                        # Can we break? No, a different 'a' might allow an even better 'b' or better net gain.
                        # But we can break the inner loop for this 'a'.
                        break 
                        # Note: We break the inner loop over b, but continue outer loop over a.
            
            if best_move is None:
                break
            
            mt = best_move[0]
            
            if mt == 'add':
                b = best_move[1]
                chosen.append(b)
                chosen_set.add(b)
                chosen_mask |= factor_masks[b]
                current_score += best_gain
            elif mt == 'swap':
                a, b = best_move[1], best_move[2]
                chosen.remove(a)
                chosen.append(b)
                chosen_set.remove(a)
                chosen_set.add(b)
                chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                current_score += best_gain
        
        return chosen, chosen_set, chosen_mask, current_score
=======
    # 2. Helper Functions
    
    def greedy_construct(ordering):
        """Greedy select in given order using bitmask coprimality."""
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask

    def fast_hill_climb(chosen, chosen_set, chosen_mask, current_score, max_rounds=200):
        """
        Best-improvement local search with 1-add, 1-swap, and 2-swap moves.
        Optimized for speed using bitmasks.
        """
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_gain = 0.0
            best_move = None  # ('add', b), ('swap', a, b), or ('2swap', a1, a2, b1, b2)
            
            # Try 1-add: add one element if possible
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                if factor_masks[b] & chosen_mask:
                    continue
                gain = scores[b]
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_move = ('add', b)
            
            # Try 1-swap: remove one, add one
            for a in range(1, 512):
                if a not in chosen_set:
                    continue
                
                temp_mask = chosen_mask & ~factor_masks[a]
                
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_move = ('swap', a, b)
                        break
            
            # Try 2-swap: remove two, add two
            # This is more expensive but can escape local optima
            # Only try if we haven't found a good 1-add or 1-swap
            if best_gain < 0.01:  # Only try 2-swap if gains are small
                chosen_list = list(chosen_set)
                n = len(chosen_list)
                
                # Try all pairs of removals (limit for speed)
                max_pairs = min(n * (n-1) // 2, 200)
                pair_count = 0
                
                for i in range(n):
                    if pair_count >= max_pairs:
                        break
                    a1 = chosen_list[i]
                    for j in range(i+1, n):
                        if pair_count >= max_pairs:
                            break
                        a2 = chosen_list[j]
                        pair_count += 1
                        
                        temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                        
                        # Find best pair of additions
                        best_pair_gain = -1.0
                        best_pair = None
                        
                        for b1 in range(511, 0, -1):
                            if b1 in chosen_set:
                                continue
                            if factor_masks[b1] & temp_mask:
                                continue
                            
                            # Check if b1 alone gives better gain than current best
                            gain1 = scores[b1] - scores[a1] - scores[a2]
                            if gain1 > best_pair_gain + 1e-12:
                                best_pair_gain = gain1
                                best_pair = (b1, None)
                            
                            # Try adding a second element b2
                            temp_mask2 = temp_mask & ~factor_masks[b1]
                            for b2 in range(511, 0, -1):
                                if b2 in chosen_set or b2 == b1:
                                    continue
                                if factor_masks[b2] & temp_mask2:
                                    continue
                                
                                gain2 = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                                if gain2 > best_pair_gain + 1e-12:
                                    best_pair_gain = gain2
                                    best_pair = (b1, b2)
                                break  # First valid b2 is best for this b1
                        
                        if best_pair and best_pair_gain > best_gain + 1e-12:
                            best_gain = best_pair_gain
                            b1, b2 = best_pair
                            if b2 is None:
                                best_move = ('2swap_add1', a1, a2, b1)
                            else:
                                best_move = ('2swap', a1, a2, b1, b2)
                            break  # Found good 2-swap, stop searching
                    if best_move and best_move[0].startswith('2swap'):
                        break
            
            if best_move is None:
                break
            
            mt = best_move[0]
            
            if mt == 'add':
                b = best_move[1]
                chosen.append(b)
                chosen_set.add(b)
                chosen_mask |= factor_masks[b]
                current_score += best_gain
            elif mt == 'swap':
                a, b = best_move[1], best_move[2]
                chosen.remove(a)
                chosen.append(b)
                chosen_set.remove(a)
                chosen_set.add(b)
                chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                current_score += best_gain
            elif mt == '2swap_add1':
                a1, a2, b1 = best_move[1], best_move[2], best_move[3]
                chosen.remove(a1)
                chosen.remove(a2)
                chosen.append(b1)
                chosen_set.remove(a1)
                chosen_set.remove(a2)
                chosen_set.add(b1)
                chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1]
                current_score += best_gain
            elif mt == '2swap':
                a1, a2, b1, b2 = best_move[1], best_move[2], best_move[3], best_move[4]
                chosen.remove(a1)
                chosen.remove(a2)
                chosen.append(b1)
                chosen.append(b2)
                chosen_set.remove(a1)
                chosen_set.remove(a2)
                chosen_set.add(b1)
                chosen_set.add(b2)
                chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                current_score += best_gain
        
        return chosen, chosen_set, chosen_mask, current_score
>>>>>>> REPLACE
