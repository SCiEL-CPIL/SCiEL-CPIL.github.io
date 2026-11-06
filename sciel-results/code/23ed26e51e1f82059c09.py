<<<<<<< SEARCH
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    # We will use a mix of random swaps and greedy swaps
    while time.time() < deadline and T > T_min:
        # Decide move type
        if rng.random() < 0.5:
            # Random Swap
            bits_set = _get_set_bits(current_mask)
            remove_idx = rng.choice(bits_set)
            add_idx = rng.randint(0, 511)
            if add_idx == remove_idx or (current_mask & (1 << add_idx)):
                continue
            new_mask = current_mask
            new_mask &= ~(1 << remove_idx)
            new_mask |= (1 << add_idx)
        else:
            # Greedy Swap: Remove random, Add best
            bits_set = _get_set_bits(current_mask)
            remove_idx = rng.choice(bits_set)
            base_mask = current_mask & ~(1 << remove_idx)
            
            best_add_score = -1
            best_add_idx = -1
            # Sample a subset of candidates to keep this fast?
            # Or scan all 512? 512 * 255 ops ~ 130k ops.
            # If we do this 100 times/sec, that's 13M ops/sec. Feasible in Python? Maybe slow.
            # Let's sample 50 candidates.
            candidates = rng.sample(range(512), 50)
            for bit in candidates:
                if base_mask & (1 << bit):
                    continue
                new_mask = base_mask | (1 << bit)
                score = _score_bitmask(new_mask)
                if score > best_add_score:
                    best_add_score = score
                    best_add_idx = bit
            
            if best_add_idx == -1:
                continue
                
            new_mask = base_mask | (1 << best_add_idx)

        new_score = _score_bitmask(new_mask)
        delta_score = new_score - current_score

        if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask

        T *= cooling
=======
    T = 1.0
    T_min = 0.01
    cooling = 0.99995
    
    # We will use a mix of random swaps and greedy swaps
    while time.time() < deadline and T > T_min:
        # Decide move type: 40% random, 60% greedy
        if rng.random() < 0.4:
            # Random Swap
            bits_set = _get_set_bits(current_mask)
            remove_idx = rng.choice(bits_set)
            add_idx = rng.randint(0, 511)
            if add_idx == remove_idx or (current_mask & (1 << add_idx)):
                continue
            new_mask = current_mask
            new_mask &= ~(1 << remove_idx)
            new_mask |= (1 << add_idx)
        else:
            # Greedy Swap: Remove random, Add best from larger sample
            bits_set = _get_set_bits(current_mask)
            remove_idx = rng.choice(bits_set)
            base_mask = current_mask & ~(1 << remove_idx)
            
            best_add_score = -1
            best_add_idx = -1
            # Sample 80 candidates for better exploration
            candidates = rng.sample(range(512), 80)
            for bit in candidates:
                if base_mask & (1 << bit):
                    continue
                new_mask = base_mask | (1 << bit)
                score = _score_bitmask(new_mask)
                if score > best_add_score:
                    best_add_score = score
                    best_add_idx = bit
            
            if best_add_idx == -1:
                continue
                
            new_mask = base_mask | (1 << best_add_idx)

        new_score = _score_bitmask(new_mask)
        delta_score = new_score - current_score

        if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask

        T *= cooling
>>>>>>> REPLACE

Now improve the final polish to be more thorough:

