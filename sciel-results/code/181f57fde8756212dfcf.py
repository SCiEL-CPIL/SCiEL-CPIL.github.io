        # Remove a random element
        ri = rng.randrange(len(current))
        removed = current[ri]
        current.pop(ri)
        cur_set.discard(removed)
        
        # Try to add two elements
        added_any = False
        for _ in range(2):
            # Try random candidates
            for _ in range(50):
                cand = rng.randrange(n_total)
                if cand in cur_set:
                    continue
                if can_add(current, cand):
                    current.append(cand)
                    cur_set.add(cand)
                    added_any = True
                    break
