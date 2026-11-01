import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Baseline: All vectors with exactly two non-zero entries (+/-1).
    # Norm^2 = 2. Max norm = sqrt(2).
    # Min distance between distinct such vectors is sqrt(2) (e.g., e1+e2 and e1+e3 -> diff e2-e3, norm sqrt(2)).
    # So all 220 are valid.
    candidates_2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_2.append(v)
    
    baseline = np.array(candidates_2, dtype=np.int64)
    best_set = baseline
    best_size = len(baseline)
    
    # Try to find a larger set from Norm^2=4 vectors.
    # If we use only Norm^2=4, max norm = 2. Need dist^2 >= 4.
    
    candidates_4 = []
    # Type A: One entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates_4.append(v)
            
    # Type B: Four entries +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates_4.append(v)
            
    candidates_4 = np.array(candidates_4, dtype=np.int64)
    n4 = len(candidates_4)
    
    if time.time() > deadline - 20:
        return best_set
    
    # Compute conflict matrix for candidates_4
    # dist^2 = n_i + n_j - 2 dot
    # All norms are 4. dist^2 = 8 - 2 dot.
    # Conflict if dist^2 < 4 => 8 - 2 dot < 4 => 2 dot > 4 => dot > 2.
    # dot is integer. So conflict if dot >= 3.
    
    dots = candidates_4 @ candidates_4.T
    conflict = (dots >= 3)
    np.fill_diagonal(conflict, False)
    
    # Precompute neighbor lists for efficiency
    neighbor_lists = [np.where(conflict[i])[0] for i in range(n4)]
    initial_degrees = np.array([len(nl) for nl in neighbor_lists], dtype=np.int32)
    
    # Greedy Independent Set with "Future Conflict Radius" heuristic
    # Strategy: pick nodes that eliminate the fewest remaining compatible partners
    np.random.seed(42)
    
    best_4_set = []
    best_4_size = 0
    
    # Number of restarts limited by time
    max_restarts = 500
    restarts_done = 0
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 3:
            break
        restarts_done += 1
        
        # Greedy with future conflict radius:
        # For each remaining node v, compute how many remaining nodes would be
        # eliminated if v is selected (i.e., |neighbors(v) ∩ remaining|)
        # Pick the node with minimum such value (break ties randomly)
        remaining = np.ones(n4, dtype=bool)
        
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Compute future conflict radius for each remaining node
            # = number of remaining nodes that conflict with it
            # This is the degree within the remaining subgraph
            fc_radii = np.zeros(len(rem_indices), dtype=np.int32)
            for idx, node in enumerate(rem_indices):
                fc_radii[idx] = conflict[node][remaining].sum()
            
            min_fc = np.min(fc_radii)
            
            # Pick one of the min-fc nodes randomly
            min_mask = fc_radii == min_fc
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            remaining[actual_remove] = False
            
            # Time check
            if time.time() > deadline - 3:
                break
        
        # Local improvement: 1-swap moves
        # Try to remove one selected node and add a different node
        improved = True
        max_swap_iterations = 50
        swap_iter = 0
        while improved and swap_iter < max_swap_iterations:
            if time.time() > deadline - 3:
                break
            improved = False
            swap_iter += 1
            
            selected_set = set(selected)
            
            for i in range(len(selected)):
                if time.time() > deadline - 3:
                    break
                    
                node_to_remove = selected[i]
                
                # Find candidates that conflict with node_to_remove
                # but don't conflict with any other selected node
                conflicts_with_removed = neighbor_lists[node_to_remove]
                
                best_candidate = -1
                best_fc_radius = np.inf
                
                for cand in conflicts_with_removed:
                    if cand in selected_set:
                        continue
                    
                    # Check if cand conflicts with any node in selected (except node_to_remove)
                    conflicts_with_new = False
                    for ns in selected:
                        if ns == node_to_remove:
                            continue
                        if conflict[cand, ns]:
                            conflicts_with_new = True
                            break
                    
                    if conflicts_with_new:
                        continue
                    
                    # Compute the future conflict radius this swap would create
                    # i.e., how many currently remaining nodes would be newly eliminated
                    # This is |neighbors(cand) ∩ remaining_after_swap|
                    # where remaining_after_swap = remaining ∪ {node_to_remove} - {cand}
                    
                    # For efficiency, just compute |neighbors(cand) ∩ (remaining ∪ {node_to_remove})|
                    # minus 1 (for cand itself if it's in that set)
                    # Actually, let's just count neighbors of cand that are in remaining or are node_to_remove
                    new_fc = 0
                    for nb in neighbor_lists[cand]:
                        if remaining[nb] or nb == node_to_remove:
                            new_fc += 1
                    
                    if new_fc < best_fc_radius:
                        best_fc_radius = new_fc
                        best_candidate = cand
                
                if best_candidate >= 0:
                    # Perform the swap
                    selected[i] = best_candidate
                    improved = True
                    break
        
        if len(selected) > best_4_size:
            best_4_size = len(selected)
            best_4_set = selected.copy()
    
    if best_4_size > best_size:
        best_set = candidates_4[best_4_set]
        best_size = best_4_size
        
    return best_set
