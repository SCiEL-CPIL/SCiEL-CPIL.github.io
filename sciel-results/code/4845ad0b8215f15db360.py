        def optimize(s, max_iters=100):
            s = list(s)
            current_score = count_collisions(s)
            s_set = set(s)
            for _ in range(max_iters):
                improved = False
                indices = list(range(12))
                random.shuffle(indices)
                for i in indices:
                    best_swap = None
                    best_score = current_score
                    orig_val = s[i]
                    # Remove orig_val from set temporarily? 
                    # No, we are checking if new_val is in s.
                    # If new_val == orig_val, skip.
                    # If new_val in s_set, skip (unless new_val == orig_val, which is handled).
                    # So we can keep s_set as is, but we must be careful.
                    # If we swap s[i] to new_val, s_set should be updated.
                    # But we are just checking validity.
                    # Valid if new_val not in s_set - {orig_val}.
                    # i.e., new_val in s_set implies new_val == orig_val or new_val is at another index.
                    # Since new_val != orig_val, new_val in s_set means it's a duplicate.
                    
                    for new_val in range(129):
                        if new_val == orig_val:
                            continue
                        if new_val in s_set:
                            continue
                        
                        s[i] = new_val
                        score = count_collisions(s)
                        if score > best_score:
                            best_score = score
                            best_swap = new_val
                            # Note: we don't update s_set here, we just record the best.
                            # We will apply it after the loop.
                    if best_swap is not None:
                        # Apply the best swap
                        s[i] = best_swap
                        # Update s_set
                        s_set.remove(orig_val)
                        s_set.add(best_swap)
                        current_score = best_score
                        improved = True
                if not improved:
                    break
            return s
    