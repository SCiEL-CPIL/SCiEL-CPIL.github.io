<<<<<<< SEARCH
    def greedy_select_vectorized(order, time_limit):
        """Greedy selection with vectorized compatibility check."""
        selected = []
        # Maintain a boolean array of which candidates are still compatible with all selected
        if not order:
            return selected

        # Start with first candidate in order
        first_idx = order[0]
        selected.append(first_idx)
        # All remaining must be compatible with first_idx
        remaining_mask = compat[first_idx].copy()  # bool array, length n
        remaining_mask[first_idx] = False  # can't select same one twice

        for idx in order[1:]:
            if time.time() > time_limit:
                break
            if not remaining_mask[idx]:
                continue
            # Check if idx is compatible with all selected: already ensured by remaining_mask
            selected.append(idx)
            # Update remaining_mask: intersect with compat[idx]
            remaining_mask = remaining_mask & compat[idx]
            remaining_mask[idx] = False

        return selected
=======
    def greedy_select_vectorized(order, time_limit):
        """Greedy selection with vectorized compatibility check."""
        selected = []
        # Maintain a boolean array of which candidates are still compatible with all selected
        if not order:
            return selected

        # Start with first candidate in order
        first_idx = order[0]
        selected.append(first_idx)
        # All remaining must be compatible with first_idx
        remaining_mask = compat[first_idx].copy()  # bool array, length n
        remaining_mask[first_idx] = False  # can't select same one twice

        for idx in order[1:]:
            if time.time() > time_limit:
                break
            if not remaining_mask[idx]:
                continue
            # Check if idx is compatible with all selected: already ensured by remaining_mask
            selected.append(idx)
            # Update remaining_mask: intersect with compat[idx]
            remaining_mask = remaining_mask & compat[idx]
            remaining_mask[idx] = False

        return selected

    def min_degree_greedy(time_limit, initial_order=None):
        """Min-degree greedy: always pick the vertex with fewest compatible neighbors among remaining."""
        if not initial_order:
            initial_order = list(range(n))
        
        selected = []
        remaining = set(initial_order)
        
        if not remaining:
            return selected
        
        # Start with the vertex of minimum degree in the full graph
        degs = {idx: compat[idx].sum() for idx in remaining}
        min_idx = min(remaining, key=lambda x: degs[x])
        selected.append(min_idx)
        remaining.remove(min_idx)
        
        # Maintain a compatibility mask for remaining vertices
        compat_mask = np.ones(n, dtype=bool)
        compat_mask[~np.isin(range(n), remaining)] = False
        
        while remaining:
            if time.time() > time_limit:
                break
            
            # For each remaining vertex, count how many other remaining vertices it's compatible with
            # This is the "degree within remaining"
            # We want to pick the one with minimum such degree
            
            # Vectorized: for each remaining idx, count compat[idx] & compat_mask
            remaining_list = list(remaining)
            if not remaining_list:
                break
            
            # Compute degrees within remaining for all remaining vertices
            # compat_mask indicates which vertices are still in remaining
            degrees_in_remaining = np.zeros(n, dtype=np.int32)
            for idx in remaining_list:
                degrees_in_remaining[idx] = np.sum(compat[idx] & compat_mask)
            
            # Find the vertex with minimum degree (tie-break by original index for determinism)
            min_deg = min(degrees_in_remaining[idx] for idx in remaining_list)
            candidates_min = [idx for idx in remaining_list if degrees_in_remaining[idx] == min_deg]
            
            # Pick one (first for determinism, or random for diversity)
            chosen = candidates_min[0]
            selected.append(chosen)
            remaining.remove(chosen)
            
            # Update compat_mask: intersect with compat[chosen]
            compat_mask = compat_mask & compat[chosen]
            compat_mask[chosen] = False
        
        return selected
>>>>>>> REPLACE

