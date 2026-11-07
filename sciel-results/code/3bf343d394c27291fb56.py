<<<<<<< SEARCH
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        fam_set = set(best_family)
        comp_counts = compute_comp_counts(best_family)
        
        for i in range(m):
            if time.time() > deadline:
                break
            
            old_mask = best_family[i]
            old_comp = comp_counts[i]
            
            best_gain = 0
            best_repl = None
            
            for s in all_masks:
                if s in fam_set:
                    continue
                
                new_comp = 0
                for j in range(m):
                    if j == i:
                        continue
                    if (s & ~best_family[j]) == 0 or (best_family[j] & ~s) == 0:
                        new_comp += 1
                
                delta = new_comp - old_comp
                if delta > best_gain:
                    best_gain = delta
                    best_repl = s
            
            if best_repl is not None and best_gain > 0:
                best_family[i] = best_repl
                fam_set.discard(old_mask)
                fam_set.add(best_repl)
                improved = True
                # Recompute comp_counts after change (or update incrementally)
                comp_counts = compute_comp_counts(best_family)
=======
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        fam_set = set(best_family)
        comp_counts = compute_comp_counts(best_family)
        
        for i in range(m):
            if time.time() > deadline:
                break
            
            old_mask = best_family[i]
            old_comp = comp_counts[i]
            
            best_gain = 0
            best_repl = None
            
            for s in all_masks:
                if s in fam_set:
                    continue
                
                new_comp = 0
                for j in range(m):
                    if j == i:
                        continue
                    if (s & ~best_family[j]) == 0 or (best_family[j] & ~s) == 0:
                        new_comp += 1
                
                delta = new_comp - old_comp
                if delta > best_gain:
                    best_gain = delta
                    best_repl = s
            
            if best_repl is not None and best_gain > 0:
                best_family[i] = best_repl
                fam_set.discard(old_mask)
                fam_set.add(best_repl)
                improved = True
                # Recompute comp_counts after change (or update incrementally)
                comp_counts = compute_comp_counts(best_family)
    
    # Final polish: try multiple random perturbations + re-local-search if time remains
    if time.time() < deadline - 2.0:
        import random as _rnd
        best_score = count_comparables(best_family)
        n_restarts = 0
        while time.time() < deadline - 1.0:
            # Create a perturbed copy
            perturbed = best_family[:]
            _rnd.shuffle(perturbed)
            # Randomly replace 3-5 elements with random masks not in the set
            pset = set(perturbed)
            for _ in range(3):
                idx = _rnd.randint(0, m - 1)
                new_mask = _rnd.randint(0, total_masks - 1)
                if new_mask not in pset:
                    pset.discard(perturbed[idx])
                    perturbed[idx] = new_mask
                    pset.add(new_mask)
            
            # Local search on perturbed
            improved_p = True
            while improved_p and time.time() < deadline - 1.0:
                improved_p = False
                fam_set_p = set(perturbed)
                comp_counts_p = compute_comp_counts(perturbed)
                for i in range(m):
                    if time.time() > deadline - 1.0:
                        break
                    old_mask_p = perturbed[i]
                    old_comp_p = comp_counts_p[i]
                    best_gain_p = 0
                    best_repl_p = None
                    for s in all_masks:
                        if s in fam_set_p:
                            continue
                        new_comp_p = 0
                        for j in range(m):
                            if j == i:
                                continue
                            if (s & ~perturbed[j]) == 0 or (perturbed[j] & ~s) == 0:
                                new_comp_p += 1
                        delta_p = new_comp_p - old_comp_p
                        if delta_p > best_gain_p:
                            best_gain_p = delta_p
                            best_repl_p = s
                    if best_repl_p is not None and best_gain_p > 0:
                        perturbed[i] = best_repl_p
                        fam_set_p.discard(old_mask_p)
                        fam_set_p.add(best_repl_p)
                        improved_p = True
                        comp_counts_p = compute_comp_counts(perturbed)
            
            score_p = count_comparables(perturbed)
            if score_p > best_score:
                best_score = score_p
                best_family = perturbed[:]
            n_restarts += 1
>>>>>>> REPLACE
