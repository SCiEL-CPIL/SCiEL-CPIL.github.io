        # Update counts and heap for r_old removal
        idx_old = np.arange(r_old, M + 1, p)
        old_counts = counts[idx_old]
        counts[idx_old] = old_counts - 1
        new_counts = counts[idx_old]
        # Became weak: was >= 2, now < 2
        became_weak = (old_counts >= 2) & (new_counts < 2)
        if np.any(became_weak):
            for x in idx_old[became_weak]:
                heapq.heappush(weak_heap, x)
                
        # Update counts and heap for r_new addition
        idx_new = np.arange(r_new, M + 1, p)
        old_counts = counts[idx_new]
        counts[idx_new] = old_counts + 1
        # No push needed for addition (can't become newly weak)
        
        new_L = get_current_L()
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = current_L - new_L
            if delta < 50 * T:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_L = new_L
                    residues[i] = r_new
                else:
                    # Revert
                    counts[idx_new] -= 1
                    # Check if idx_new became weak? No, subtraction makes it weaker.
                    # Wait, we added then subtracted. 
                    # Original count C. Added -> C+1. Subtracted -> C.
                    # If C < 2, it was weak originally.
                    # Did we push it? 
                    # In forward move (addition), we didn't push.
                    # So if it is weak now (C < 2), it might not be in heap.
                    # We should push it.
                    new_counts_revert = counts[idx_new]
                    if np.any(new_counts_revert < 2):
                        for x in idx_new[new_counts_revert < 2]:
                            heapq.heappush(weak_heap, x)
                            
                    counts[idx_old] += 1
                    # Addition, no push.
            else:
                # Revert
                counts[idx_new] -= 1
                new_counts_revert = counts[idx_new]
                if np.any(new_counts_revert < 2):
                    for x in idx_new[new_counts_revert < 2]:
                        heapq.heappush(weak_heap, x)
                counts[idx_old] += 1
