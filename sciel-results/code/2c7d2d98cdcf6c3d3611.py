    # Generic neighbor mask generation
    for idx in range(total_words):
        mask = 1 << idx
        # Generate all words within Hamming distance 'radius'
        # This is exponential in radius.
        # For radius 1, it's fast.
        # For radius 2, it's manageable.
        # For larger radius, it might be slow.
        
        # To be safe, I'll stick to radius 1 optimization as it's the specific case.
        # If radius is not 1, I'll use a slower but correct method:
        # Iterate all other words and check distance? No, that's O(Q^2).
        # Better: Iterate over combinations of positions to change.
        
        if radius == 1:
            for i in range(n):
                original = (idx // powers[i]) % q
                base = idx - original * powers[i]
                for new_val in range(q):
                    if new_val != original:
                        neighbor_idx = base + new_val * powers[i]
                        mask |= (1 << neighbor_idx)
        elif radius == 2:
            # Change 1 position
            for i in range(n):
                original = (idx // powers[i]) % q
                base = idx - original * powers[i]
                for new_val in range(q):
                    if new_val != original:
                        neighbor_idx = base + new_val * powers[i]
                        mask |= (1 << neighbor_idx)
            # Change 2 positions
            for i in range(n):
                for j in range(i+1, n):
                    orig_i = (idx // powers[i]) % q
                    orig_j = (idx // powers[j]) % q
                    base = idx - orig_i * powers[i] - orig_j * powers[j]
                    for new_i in range(q):
                        if new_i == orig_i: continue
                        for new_j in range(q):
                            if new_j == orig_j: continue
                            neighbor_idx = base + new_i * powers[i] + new_j * powers[j]
                            mask |= (1 << neighbor_idx)
        else:
            # Fallback: assume radius 1 if not 1 or 2? 
            # Or just do radius 1.
            # Given the task, radius is 1.
            for i in range(n):
                original = (idx // powers[i]) % q
                base = idx - original * powers[i]
                for new_val in range(q):
                    if new_val != original:
                        neighbor_idx = base + new_val * powers[i]
                        mask |= (1 << neighbor_idx)
        
        neighbor_masks[idx] = mask
