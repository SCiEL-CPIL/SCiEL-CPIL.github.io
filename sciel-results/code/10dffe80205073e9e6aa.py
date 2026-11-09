<<<<<<< SEARCH
def _local_search_improve(S_mask, primes, start_time, deadline):
    """Try to add back removed elements to improve the set size while maintaining admissibility."""
    S_mask = S_mask.copy()
    improved = True
    
    while improved and time.time() - start_time < deadline:
        improved = False
        
        # Get current size and check which primes are constraints
        current_size = np.count_nonzero(S_mask)
        if current_size == 0:
            break
        
        # Find all removed elements
        removed = np.where(~S_mask)[0]
        if len(removed) == 0:
            break
        
        # Try adding back each removed element
        for val in removed:
            if time.time() - start_time > deadline:
                break
            
            # Try adding this element
            S_mask[val] = True
            new_size = np.count_nonzero(S_mask)
            
            # Check if still admissible
            if _is_admissible(S_mask, primes):
                improved = True
                current_size = new_size
            else:
                S_mask[val] = False
    
    return S_mask
=======
def _local_search_improve(S_mask, primes, start_time, deadline):
    """
    Improved local search: try to add back removed elements, prioritizing
    those with lowest conflict degree. Multiple passes until no improvement.
    """
    S_mask = S_mask.copy()
    
    # Compute initial conflict degrees
    def compute_degrees(mask):
        deg = np.zeros(N + 1, dtype=np.int32)
        S_idx = np.where(mask)[0]
        if len(S_idx) == 0:
            return deg
        max_p = max(len(S_idx), 50)
        relevant = primes[primes <= max_p]
        for p in relevant:
            res = S_idx % p
            cnt = np.bincount(res, minlength=p)
            deg[S_idx] += cnt[res] - 1
        return deg
    
    improved = True
    max_passes = 5
    
    for _ in range(max_passes):
        if time.time() - start_time > deadline:
            break
        if not improved:
            break
        improved = False
        
        current_size = np.count_nonzero(S_mask)
        if current_size == 0:
            break
        
        removed = np.where(~S_mask)[0]
        if len(removed) == 0:
            break
        
        # Compute degrees for removed elements
        deg = compute_degrees(S_mask)
        removed_deg = deg[removed]
        
        # Sort by degree ascending (prefer low-conflict additions)
        order = np.argsort(removed_deg)
        removed_sorted = removed[order]
        
        for val in removed_sorted:
            if time.time() - start_time > deadline:
                break
            
            S_mask[val] = True
            if _is_admissible(S_mask, primes):
                improved = True
            else:
                S_mask[val] = False
    
    return S_mask
>>>>>>> REPLACE
