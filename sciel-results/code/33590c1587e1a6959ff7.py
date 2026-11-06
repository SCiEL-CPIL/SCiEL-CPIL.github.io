<<<<<<< SEARCH
    # Phase 2: If we didn't get a full solution, try greedy + local search
    if best_len < len(all_mods):
        for ordering in orderings:
            if time.time() > deadline - 5.0:
                break
            for seed in range(20):
                if time.time() > deadline - 2.0:
                    break
                pairs = greedy_construct(ordering, deadline - 1.0, seed)
                if len(pairs) > best_len:
                    best_pairs = pairs
                    best_len = len(pairs)
                
                # Local search refinement
                if time.time() < deadline - 1.0:
                    refined = local_search_refine(pairs, deadline - 0.5, seed)
                    if len(refined) > best_len:
                        best_pairs = refined
                        best_len = len(refined)
=======
    # Phase 2: If we didn't get a full solution, try greedy + local search
    if best_len < len(all_mods):
        for ordering in orderings:
            if time.time() > deadline - 5.0:
                break
            for seed in range(20):
                if time.time() > deadline - 2.0:
                    break
                pairs = greedy_construct(ordering, deadline - 1.0, seed)
                if len(pairs) > best_len:
                    best_pairs = pairs
                    best_len = len(pairs)
                
                # Swap local search refinement
                if time.time() < deadline - 1.0:
                    refined = swap_local_search(pairs, deadline - 0.5, seed)
                    if len(refined) > best_len:
                        best_pairs = refined
                        best_len = len(refined)
>>>>>>> REPLACE
