def construct():
    # Beam search for Maximum Length Sequence
    # State: (values_list, products_set)
    beam = [([], set())]
    beam_size = 2000
    
    for x in range(2, 129):
        new_beam = []
        for values, products in beam:
            # Option 1: Skip x
            new_beam.append((values, products))
            
            # Option 2: Include x
            # Calculate products of blocks ending at x
            tails = [x]
            prod = x
            # Iterate through existing values in reverse to build products
            # values is strictly increasing, so reversed is decreasing
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            
            # Check for collisions
            # If any tail product is already in the set of products, collision
            if not products.intersection(tails):
                new_products = products | set(tails)
                new_values = values + [x]
                new_beam.append((new_values, new_products))
        
        # Prune the beam
        # Sort by length descending, then by number of products ascending (fewer products = more flexible)
        new_beam.sort(key=lambda s: (len(s[0]), -len(s[1])), reverse=True)
        
        # Deduplicate based on values
        seen = set()
        pruned_beam = []
        for state in new_beam:
            key = tuple(state[0])
            if key not in seen:
                seen.add(key)
                pruned_beam.append(state)
            if len(pruned_beam) >= beam_size:
                break
        
        beam = pruned_beam
        
    if not beam:
        return []
    
    # Return the longest sequence
    best_state = max(beam, key=lambda s: len(s[0]))
    return best_state[0]
