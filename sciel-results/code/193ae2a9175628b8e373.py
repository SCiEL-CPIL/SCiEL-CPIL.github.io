import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based greedy construction, effective local search, and a branch-and-bound
    Maximum Independent Set solver with greedy coloring upper bound for exact optimization
    on the remaining candidates.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before the 630s kill

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    forb_list = list(forbidden)
    forb_arr = np.array(forb_list)

    # Build conflict masks
    # conflict_mask[i] is an integer where bit j is set if i conflicts with j
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask

    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            # Check if candidate conflicts with any selected
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def local_search(subset, mask):
        """
        Local search: try to increase size by removing elements and greedily adding back.
        Returns improved subset and mask.
        """
        subset = list(subset)
        mask = mask
        improved = True
        while improved:
            improved = False
            n = len(subset)
            
            # Try removing 1 element and greedily adding back
            for i in range(n):
                if time.time() > deadline - 5:
                    return subset, mask
                u = subset[i]
                # Remove u
                trial_mask = mask & ~(1 << u)
                trial_subset = subset[:i] + subset[i+1:]
                
                # Greedily add back compatible elements
                new_mask = trial_mask
                new_subset = list(trial_subset)
                for v in range(m):
                    if (new_mask & (1 << v)) == 0 and (conflict_mask[v] & new_mask) == 0:
                        new_mask |= (1 << v)
                        new_subset.append(v)
                
                if len(new_subset) > len(subset):
                    subset = new_subset
                    mask = new_mask
                    improved = True
                    break
            
            # Try removing 2 elements and greedily adding back
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5:
                        return subset, mask
                    for j in range(i+1, n):
                        if time.time() > deadline - 5:
                            return subset, mask
                        u, v = subset[i], subset[j]
                        # Remove u and v
                        trial_mask = mask & ~(1 << u) & ~(1 << v)
                        trial_subset = [x for x in subset if x != u and x != v]
                        
                        # Greedily add back compatible elements
                        new_mask = trial_mask
                        new_subset = list(trial_subset)
                        for w in range(m):
                            if (new_mask & (1 << w)) == 0 and (conflict_mask[w] & new_mask) == 0:
                                new_mask |= (1 << w)
                                new_subset.append(w)
                        
                        if len(new_subset) > len(subset):
                            subset = new_subset
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break
        
        return subset, mask

    def branch_and_bound(candidates, current_size, current_mask):
        """
        Branch and bound for Maximum Independent Set.
        candidates: list of vertices to consider.
        current_size: size of the independent set found so far.
        current_mask: bitmask of vertices already selected.
        Returns the best independent set (list) and its mask found within time limit.
        """
        best_set = []
        best_mask = current_mask
        best_size = current_size
        
        # Precompute degrees within the induced subgraph of candidates
        # To speed up coloring, we can just use the conflict masks restricted to candidates
        
        def color_bound(cands):
            """
            Greedy coloring of the graph induced by cands.
            Returns the number of colors used (upper bound on MIS size in cands).
            """
            if not cands:
                return 0
            
            colors = {}
            num_colors = 0
            uncolored = list(cands)
            
            while uncolored:
                num_colors += 1
                # Greedy coloring: assign current color to as many vertices as possible
                color_class = []
                new_uncolored = []
                
                # Order by degree descending for better coloring (optional optimization)
                # For now, just take them in order
                for v in uncolored:
                    # Check if v conflicts with any vertex in color_class
                    conflict_with_class = False
                    for u in color_class:
                        if (conflict_mask[v] >> u) & 1:
                            conflict_with_class = True
                            break
                    
                    if not conflict_with_class:
                        color_class.append(v)
                        colors[v] = num_colors
                    else:
                        new_uncolored.append(v)
                
                uncolored = new_uncolored
            
            return num_colors

        def solve(cands, size, mask):
            nonlocal best_set, best_mask, best_size
            
            if not cands:
                return
            
            if size >= best_size:
                # Update best if strictly better or same size (to keep finding valid ones)
                if size > best_size:
                    best_size = size
                    best_mask = mask
                    # Reconstruct set from mask for candidates? 
                    # Actually, we need to track the set. 
                    # Let's pass the set too or reconstruct.
                    # Reconstructing from mask is O(m) which is fine.
                    best_set = [i for i in range(m) if (mask >> i) & 1]
            
            # Upper bound: size + color_bound(cands)
            if size + color_bound(cands) <= best_size:
                return
            
            if time.time() > deadline - 2:
                return

            # Choose a pivot vertex
            # Heuristic: vertex with max degree in the induced subgraph
            pivot = cands[0]
            max_deg = -1
            for v in cands:
                deg = 0
                for u in cands:
                    if v != u and ((conflict_mask[v] >> u) & 1):
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            
            # Branch 1: Include pivot
            # New candidates are neighbors of pivot? No, non-neighbors (compatible vertices).
            # If we include pivot, we must exclude all its neighbors.
            # The remaining candidates are those in cands that are compatible with pivot.
            new_cands_include = [v for v in cands if v != pivot and not ((conflict_mask[pivot] >> v) & 1)]
            solve(new_cands_include, size + 1, mask | (1 << pivot))
            
            # Branch 2: Exclude pivot
            new_cands_exclude = [v for v in cands if v != pivot]
            solve(new_cands_exclude, size, mask)

        # Initial candidates are those not in current_mask
        # We want to find an independent set in the graph induced by ~current_mask
        # But we already have a valid set in current_mask. We are looking for additions.
        # So the "graph" is the subgraph of vertices NOT in current_mask.
        # We want to find an MIS in this subgraph.
        
        # However, the standard B&B for MIS usually starts with all vertices.
        # Here, we already have a base set. We just want to add more.
        # The vertices available are those not in current_mask.
        # Let's filter candidates to only those not in current_mask.
        
        avail = [v for v in range(m) if not (current_mask >> v) & 1]
        
        # Optimization: If avail is empty, return
        if not avail:
            return [], current_mask
        
        # Run B&B
        solve(avail, 0, 0) # Size 0 relative to the additions
        
        # The best_set returned is the additions.
        # We need to combine with current_mask?
        # Wait, solve updates best_mask which is relative to the additions.
        # Let's adjust the logic.
        
        # Actually, let's just return the best additions found.
        # The caller will combine them.
        
        # Note: The solve function above updates best_mask to be the mask of the ADDITIONS.
        # So best_set is the list of added vertices.
        
        return best_set, best_mask

    best_subset = []
    best_size = 0
    best_mask = 0
    
    random.seed(42)
    
    # Run many random greedy trials with local search
    for trial in range(5000):
        if time.time() > deadline - 20:
            break
            
        order = list(range(m))
        random.shuffle(order)
        
        subset, mask = greedy_construct(order)
        
        # Apply local search to try to improve
        subset, mask = local_search(subset, mask)
        
        # Try to improve further with Branch and Bound on the remaining vertices
        # Only do B&B if we have time and if the set is not too large (B&B is exponential)
        # If the set is already large, the remaining graph is small, B&B is fast.
        if time.time() < deadline - 30:
            # Check if we can find more
            # If the current size is high, the complement is small, B&B is efficient
            if m - len(subset) < 40: # Heuristic: only B&B if few vertices left
                additions, add_mask = branch_and_bound(None, 0, mask)
                if additions:
                    new_subset = subset + additions
                    new_mask = mask | add_mask
                    if len(new_subset) > len(subset):
                        subset = new_subset
                        mask = new_mask
        
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
            best_mask = mask
    
    # Final check
    if not best_subset:
        # Fallback
        subset, mask = greedy_construct(list(range(m)))
        best_subset = subset
        best_mask = mask
        best_size = len(best_subset)

    return np.array(best_subset, dtype=np.int64)
