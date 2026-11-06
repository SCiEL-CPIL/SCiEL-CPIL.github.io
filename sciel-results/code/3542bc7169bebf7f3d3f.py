<<<<<<< SEARCH
    # Phase 2: Local Search on top candidates
    search_start = time.time()
    
    for e, chain in top_candidates:
        if time.time() - search_start > time_for_search:
            break
        if _time_left() < 15:
            break
            
        remaining_search = time_for_search - (time.time() - search_start)
        if remaining_search < 0.5:
            break
            
        improved = _local_search(sequence, chain, remaining_search * 0.9, rng)
        
        if improved:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy or best_chain is None:
                best_energy = e_imp
                best_chain = improved
=======
    # Phase 2: Local Search on top candidates
    search_start = time.time()
    
    for e, chain in top_candidates:
        if time.time() - search_start > time_for_search:
            break
        if _time_left() < 15:
            break
            
        remaining_search = time_for_search - (time.time() - search_start)
        if remaining_search < 0.5:
            break
            
        improved = _local_search(sequence, chain, remaining_search * 0.9, rng)
        
        if improved:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy or best_chain is None:
                best_energy = e_imp
                best_chain = improved
    
    # Phase 2b: Additional MCTS refinement if time permits
    if _time_left() > 30 and best_chain is not None:
        refine_start = time.time()
        refine_time = min(10.0, _time_left() - 30)
        
        # Use the best chain as a starting point for more MCTS exploration
        for i in range(3):
            if time.time() - refine_start > refine_time:
                break
            if _time_left() < 20:
                break
            
            # Perturb the best chain slightly and run MCTS
            perturbed = list(best_chain)
            # Randomly swap a few segments
            for _ in range(2):
                if len(perturbed) > 5:
                    idx = rng.randint(2, len(perturbed) - 3)
                    # Try to flip a small segment
                    segment = perturbed[idx:idx+2]
                    if len(segment) == 2:
                        # Rotate the segment
                        px, py = perturbed[idx-1]
                        new_seg = []
                        for p in segment:
                            dx, dy = p[0] - px, p[1] - py
                            # 90 degree rotation
                            new_seg.append((px - dy, py + dx))
                        # Check validity
                        valid = True
                        for j, p in enumerate(new_seg):
                            pos_idx = idx + j
                            if p in set(perturbed[:pos_idx]) | set(perturbed[pos_idx+1:]):
                                valid = False
                                break
                        if valid:
                            perturbed[idx:idx+2] = new_seg
            
            chain = _mcts_grow(sequence, rng, refine_time / 3)
            if chain:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain
>>>>>>> REPLACE

