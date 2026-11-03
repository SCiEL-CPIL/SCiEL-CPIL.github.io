def try_exchange(solution_set, cover_count, subsets_of_block, blocks_of_subset):
    changed = True
    while changed:
        changed = False
        for b_idx in list(solution_set):
            subs_b = subsets_of_block[b_idx]
            # Find subsets covered ONLY by b_idx
            critical = [s for s in subs_b if cover_count[s] == 1]
            
            if not critical:
                # Remove
                solution_set.remove(b_idx)
                for s in subs_b:
                    cover_count[s] -= 1
                changed = True
            else:
                # Try to find replacement
                # Intersect blocks_of_subset for all critical subsets
                # Optimization: Start with the critical subset that has fewest covering blocks
                critical_sorted = sorted(critical, key=lambda s: len(blocks_of_subset[s]))
                candidate_blocks = set(blocks_of_subset[critical_sorted[0]])
                
                for s in critical_sorted[1:]:
                    candidate_blocks &= set(blocks_of_subset[s])
                    if not candidate_blocks:
                        break
                
                # Filter out blocks already in solution
                valid_candidates = [b for b in candidate_blocks if b not in solution_set]
                
                if valid_candidates:
                    # Pick the one that covers the most NEW subsets?
                    # Or just pick the first one to keep it fast?
                    # Let's pick the one with max new coverage to be safe.
                    best_new = -1
                    best_gain = -1
                    for b_new in valid_candidates:
                        gain = 0
                        for s in subsets_of_block[b_new]:
                            if cover_count[s] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_new = b_new
                    
                    if best_new != -1:
                        # Perform swap
                        solution_set.remove(b_idx)
                        for s in subs_b:
                            cover_count[s] -= 1
                        # Note: cover_count for critical subsets becomes 0, then we add b_new
                        # Actually, we should update counts properly.
                        # Remove b_idx
                        # Add b_new
                        solution_set.add(best_new)
                        for s in subsets_of_block[best_new]:
                            cover_count[s] += 1
                        changed = True
                        break # Restart loop after change? Or continue? 
                        # Breaking and restarting the while loop is safer for correctness of cover_count updates if multiple swaps happen.
