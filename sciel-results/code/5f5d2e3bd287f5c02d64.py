# EVOLVE-BLOCK-START
def construct():
    """
    Returns a strictly increasing sequence of integers in [2, 128] 
    maximizing length such that all contiguous block products are distinct.
    Uses Beam Search with prime exponent vectors.
    """
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    factor_vecs = {}
    for n in range(2, 129):
        n_tmp = n
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n_tmp % p == 0:
                vec[i] += 1
                n_tmp //= p
            if n_tmp == 1:
                break
        factor_vecs[n] = tuple(vec)
    
    zero_vec = tuple([0] * num_primes)
    
    # State: (values, prefix_vecs, product_set)
    # values: list of ints
    # prefix_vecs: list of tuples, prefix_vecs[0] = 0, prefix_vecs[i] = sum(factor_vecs[values[0:i]])
    # product_set: set of tuples, all contiguous block products
    
    def add_value(state, x):
        """
        Try to add x to the state.
        Returns new state or None if invalid.
        """
        values, prefix_vecs, product_set = state
        k = len(values)
        
        # New prefix
        last_prefix = prefix_vecs[-1]
        new_prefix = tuple(last_prefix[i] + factor_vecs[x][i] for i in range(num_primes))
        
        # Generate new block products ending at x
        new_products = []
        # prefix_vecs has length k+1 (indices 0..k)
        # New blocks are prefix_vecs_new[-1] - prefix_vecs[i] for i in 0..k
        # prefix_vecs_new[-1] is new_prefix
        for i in range(k + 1):
            old_prefix = prefix_vecs[i]
            block_vec = tuple(new_prefix[j] - old_prefix[j] for j in range(num_primes))
            if block_vec in product_set:
                return None
            new_products.append(block_vec)
        
        # If no collision, update state
        new_product_set = product_set.copy()
        new_product_set.update(new_products)
        
        new_values = values + [x]
        new_prefix_vecs = prefix_vecs + [new_prefix]
        
        return (new_values, new_prefix_vecs, new_product_set)
    
    # Beam Search
    beam_width = 500
    # Initial state: empty sequence
    # values: []
    # prefix_vecs: [zero_vec]
    # product_set: set()
    current_beam = [([], [zero_vec], set())]
    best_len = 0
    
    for x in range(2, 129):
        next_candidates = []
        
        for state in current_beam:
            # Option 1: Skip x
            next_candidates.append(state)
            
            # Option 2: Add x
            new_state = add_value(state, x)
            if new_state is not None:
                next_candidates.append(new_state)
        
        # Prune based on potential
        # Max possible length from this point is len(state) + (128 - x)
        # If this is <= best_len, discard
        # Note: best_len is updated after filtering? Or before?
        # Let's filter first, then update best_len from the survivors.
        
        # First pass: filter out hopeless states
        filtered_candidates = []
        for s in next_candidates:
            potential = len(s[0]) + (128 - x)
            if potential > best_len:
                filtered_candidates.append(s)
        
        if not filtered_candidates:
            # If all are hopeless, we can break? No, maybe best_len is stale?
            # But if potential <= best_len for ALL, then no state can improve.
            # However, we still need to return the best state found so far.
            # The best state is in current_beam or previous best.
            # Actually, if we prune everything, we lost the states.
            # We should keep at least the best state from current_beam?
            # No, if potential <= best_len, it means even if we add all remaining numbers, we won't beat best_len.
            # So we can stop searching?
            # But we need to return a sequence.
            # We can break and return the best state seen so far.
            # Let's store best_state globally.
            pass

        # Update best_len
        if filtered_candidates:
            current_max = max(len(s[0]) for s in filtered_candidates)
            if current_max > best_len:
                best_len = current_max
        
        # Sort by length descending, then by sum of values descending (prefer larger/safer numbers)
        # Using sum as tie-breaker
        filtered_candidates.sort(key=lambda s: (-len(s[0]), -sum(s[0])))
        
        # Keep top beam_width
        current_beam = filtered_candidates[:beam_width]
        
        # If current_beam is empty, we should probably revert to previous best?
        # But the logic above should prevent it if best_len is tracked correctly.
        # However, if best_len is 0 initially, and we prune everything?
        # Initial state: len 0. potential 127. best_len 0. 127 > 0. Keeps.
    
    # Find best state
    if current_beam:
        best_state = max(current_beam, key=lambda s: len(s[0]))
        return best_state[0]
    else:
        # Fallback
        return []

# EVOLVE-BLOCK-END
