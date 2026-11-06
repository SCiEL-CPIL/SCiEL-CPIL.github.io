<<<<<<< SEARCH
    # Phase 2: Local Search on best_system
    if best_system and time.time() < deadline - 2:
        current = best_system[:]
        current_alpha = best_alpha
        used_pairs = set()
        for t in current:
            used_pairs.update(triple_pairs[t])
            
        # Precompute valid triples for quick check?
        # Just check linearity on the fly.
        
        while time.time() < deadline - 2:
            # Pick a random triple to remove
            idx = random.randint(0, len(current) - 1)
            t_out = current[idx]
            
            # Remove t_out
            current.pop(idx)
            for p in triple_pairs[t_out]:
                used_pairs.discard(p)
                
            # Find a triple to add
            # Try random triples from all_triples
            # To make it faster, we can pre-filter triples that don't conflict with used_pairs
            # But used_pairs changes.
            
            # Simple approach: try random triples until one fits
            added = False
            for _ in range(100): # Try 100 random candidates
                t_in = random.choice(all_triples)
                if t_in in current:
                    continue
                pairs_in = triple_pairs[t_in]
                if not any(p in used_pairs for p in pairs_in):
                    current.append(t_in)
                    used_pairs.update(pairs_in)
                    added = True
                    break
            
            if not added:
                # Revert
                current.insert(idx, t_out)
                for p in triple_pairs[t_out]:
                    used_pairs.add(p)
                continue
            
            # Evaluate new state
            new_alpha = calc_alpha(current)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_system = current[:]
            else:
                # Revert
                t_in = current.pop()
                for p in triple_pairs[t_in]:
                    used_pairs.discard(p)
                current.insert(idx, t_out)
                for p in triple_pairs[t_out]:
                    used_pairs.add(p)

    return best_system
=======
    # Phase 2: Local Search on best_system
    if best_system and time.time() < deadline - 1:
        current = best_system[:]
        current_alpha = best_alpha
        used_pairs = set()
        for t in current:
            used_pairs.update(triple_pairs[t])
            
        while time.time() < deadline - 1:
            if not current:
                break
                
            # Pick a random triple to remove
            idx = random.randint(0, len(current) - 1)
            t_out = current[idx]
            
            # Remove t_out
            current.pop(idx)
            for p in triple_pairs[t_out]:
                used_pairs.discard(p)
                
            # Find a triple to add
            added = False
            for _ in range(20): # Reduced candidates per step, more steps overall
                t_in = random.choice(all_triples)
                if t_in in current:
                    continue
                pairs_in = triple_pairs[t_in]
                if not any(p in used_pairs for p in pairs_in):
                    current.append(t_in)
                    used_pairs.update(pairs_in)
                    added = True
                    break
            
            if not added:
                # Revert
                current.insert(idx, t_out)
                for p in triple_pairs[t_out]:
                    used_pairs.add(p)
                continue
            
            # Evaluate new state
            new_alpha = calc_alpha_fast(current)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_system = current[:]
            else:
                # Revert
                t_in = current.pop()
                for p in triple_pairs[t_in]:
                    used_pairs.discard(p)
                current.insert(idx, t_out)
                for p in triple_pairs[t_out]:
                    used_pairs.add(p)

    return best_system
>>>>>>> REPLACE
