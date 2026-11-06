def _local_search_refinement(best_set, all_numbers, deadline_check):
    """
    Try to improve the set by adding elements or performing swaps.
    """
    final_set = set(best_set)
    remaining = [v for v in all_numbers if v not in final_set]
    remaining.sort(reverse=True)
    
    improved = True
    while improved and not deadline_check():
        improved = False
        
        # 1. Try to add a single element
        for v in remaining:
            if deadline_check():
                break
            trial = final_set | {v}
            if _is_reciprocal_subsum_free(list(trial)):
                final_set = trial
                remaining.remove(v)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try to swap: remove u, add v (size remains same)
        # This helps escape local optima where no single add works.
        found_swap = False
        for u in list(final_set):
            if deadline_check():
                break
            for v in remaining:
                if deadline_check():
                    break
                trial = (final_set - {u}) | {v}
                if _is_reciprocal_subsum_free(list(trial)):
                    final_set = trial
                    remaining.remove(v)
                    remaining.append(u)
                    remaining.sort(reverse=True)
                    found_swap = True
                    improved = True
                    break
            if found_swap:
                break
        
        if improved:
            continue

        # 3. Try to remove 1, add 2 (size increases by 1)
        found_expand = False
        for u in list(final_set):
            if deadline_check():
                break
            # Try pairs from remaining
            # Optimization: only check pairs if we have time
            for i in range(len(remaining)):
                if deadline_check():
                    break
                for j in range(i+1, len(remaining)):
                    if deadline_check():
                        break
                    trial = (final_set - {u}) | {remaining[i], remaining[j]}
                    if _is_reciprocal_subsum_free(list(trial)):
                        final_set = trial
                        # Update remaining
                        remaining.remove(remaining[i])
                        remaining.remove(remaining[j])
                        remaining.append(u)
                        remaining.sort(reverse=True)
                        found_expand = True
                        improved = True
                        break
                if found_expand:
                    break
            if found_expand:
                break
                
    return final_set
