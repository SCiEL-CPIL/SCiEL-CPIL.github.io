<<<<<<< SEARCH
    # Final Deletion Pass
    if best_solution is not None and time.time() < deadline - 2:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
                    
            best_solution = new_solution
            best_size = len(best_solution)
=======
    # Final Deletion Pass (Enhanced)
    if best_solution is not None and time.time() < deadline - 2:
        # Multiple passes to ensure no redundant blocks remain
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in best_solution:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
                    
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                new_solution = []
                for b_idx in best_solution:
                    subs_b = subsets_of_block[b_idx]
                    can_remove = True
                    for s_idx in subs_b:
                        if cover_count[s_idx] == 1:
                            can_remove = False
                            break
                            
                    if can_remove:
                        cover_count[subs_b] -= 1
                        changed = True
                    else:
                        new_solution.append(b_idx)
                        
                best_solution = new_solution
                best_size = len(best_solution)
>>>>>>> REPLACE
