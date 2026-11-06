import random
import itertools
import time

def construct():
    n = 16
    points = list(range(n))
    full_mask = (1 << n) - 1
    
    # Precompute popcount for all 16-bit masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Precompute all triples as bitmasks
    all_triples = []
    for comb in itertools.combinations(points, 3):
        m = 0
        for v in comb:
            m |= (1 << v)
        all_triples.append(m)
        
    # Map mask to list of points for final output
    mask_to_points = {}
    for comb in itertools.combinations(points, 3):
        m = 0
        for v in comb:
            m |= (1 << v)
        mask_to_points[m] = list(comb)

    def get_largest_independent_set(edge_masks):
        """
        Returns the mask of the largest independent set.
        An independent set is a set of vertices containing no edge.
        """
        if not edge_masks:
            return full_mask
            
        # is_indep[i] is True if i is an independent set
        is_indep = [True] * (1 << n)
        
        # Mark all supersets of edges as non-independent
        for em in edge_masks:
            complement = (~em) & full_mask
            sub = complement
            while True:
                s = em | sub
                is_indep[s] = False
                if sub == 0:
                    break
                sub = (sub - 1) & complement
                
        # Find the largest independent set
        max_size = 0
        best_mask = 0
        # Iterate through all subsets
        # Optimization: We can iterate by size descending to stop early?
        # Or just iterate all and track max. 65536 is small.
        for i in range(1 << n):
            if is_indep[i]:
                sz = popcount[i]
                if sz > max_size:
                    max_size = sz
                    best_mask = i
                    if max_size == n:
                        break
                        
        return best_mask

    def is_linear(edge_mask, existing_masks):
        for em in existing_masks:
            if popcount[edge_mask & em] > 1:
                return False
        return True

    best_E_masks = []
    best_alpha = n
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    # Heuristic Search:
    # We try to construct a hypergraph where alpha <= 4.
    # If we fail, we keep the best found.
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        E_masks = []
        
        # Heuristic: Always try to hit the current largest independent set
        while True:
            # Check if we already have a good solution
            # We need to calculate alpha. 
            # To save time, we only calculate the *largest* independent set mask, 
            # which gives us the alpha size directly.
            largest_ind_mask = get_largest_independent_set(E_masks)
            current_alpha = popcount[largest_ind_mask]
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_E_masks = E_masks[:]
                
            if current_alpha <= 4:
                # Found a very good solution
                return [mask_to_points[m] for m in E_masks]
                
            # If current_alpha is 5 or 6, we try to break it.
            # If we are stuck (maximal), we break.
            
            # Find candidate triples that intersect the largest independent set
            candidates = []
            # We can restrict candidates to those intersecting largest_ind_mask
            # to ensure we reduce alpha.
            
            for t_mask in all_triples:
                # Check if it intersects the independent set
                if (t_mask & largest_ind_mask) == 0:
                    continue
                
                # Check linearity
                if is_linear(t_mask, E_masks):
                    # Score candidate: prefer those hitting more of the independent set
                    hits = popcount[t_mask & largest_ind_mask]
                    candidates.append((hits, t_mask))
                    
            if not candidates:
                # Maximal hypergraph reached, but alpha > 4
                break
                
            # Select a candidate
            # Prefer higher hits (3 > 2 > 1)
            # Randomize within the top group to allow exploration
            max_hits = max(c[0] for c in candidates)
            top_candidates = [c[1] for c in candidates if c[0] == max_hits]
            
            # Add some randomness: sometimes pick from slightly lower hits to avoid local optima?
            # For now, stick to top hits for directed search.
            chosen_mask = random.choice(top_candidates)
            E_masks.append(chosen_mask)
            
            # If we have added many triples, check time
            if time.time() - start_time > time_limit:
                break
                
    # If we didn't find alpha <= 4, return the best found
    # Ensure we return a valid list of lists
    return [mask_to_points[m] for m in best_E_masks]
