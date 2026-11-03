"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _generate_seed_vectors():
    """Generate a small valid set as a fallback."""
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # We focus on shell m=4.
    # Norm squared = 4. Max norm M = 2.
    # Constraint: ||u - v|| >= 2 => ||u-v||^2 >= 4.
    # ||u-v||^2 = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4 => 4 >= 2 u.v => u.v <= 2.
    # So we need a subset of shell 4 where all pairwise inner products are <= 2.
    
    m = 4
    max_ip_allowed = 2
    
    candidates = _generate_shell(m)
    N = len(candidates)
    
    # Precompute the Gram matrix (inner products) for all pairs.
    # N=5280. 5280x5280 matrix of int8/int16.
    # 5280^2 ~ 28M entries. int8 is enough since max IP is 4.
    # 28MB memory is fine.
    # Computing it: candidates @ candidates.T.
    # 5280 x 11 dot 11 x 5280. Fast in BLAS.
    
    start_time = time.time()
    G = candidates @ candidates.T
    # G[i, j] is inner product of candidates[i] and candidates[j].
    # We need G[i, j] <= max_ip_allowed for all i != j in selected set.
    # G[i, i] = 4, which is > 2, so we must exclude self-pairs or handle them.
    # Since we select distinct indices, self-pairs don't matter for validity, 
    # but we must ensure we don't pick the same vector twice.
    
    # Simulated Annealing
    # State: boolean mask 'selected' of length N.
    # Objective: Maximize sum(selected).
    # Constraint: For all i, j where selected[i] and selected[j] and i!=j, G[i,j] <= max_ip_allowed.
    
    # To make SA efficient, we need to quickly check if adding/removing a vector violates constraints.
    # Let's maintain an array 'conflict_count' or similar?
    # Actually, checking validity of a move is O(N) if we just check the moved element against all others.
    # Move types:
    # 1. Add a random unselected vector v.
    #    Check: For all u in selected, G[v, u] <= max_ip_allowed.
    #    If valid, accept (improvement).
    # 2. Remove a random selected vector v.
    #    Always valid (improves or maintains constraint, decreases score).
    # 3. Swap: Remove v_in, Add v_out.
    #    Check: 
    #    a) v_out is compatible with all (selected \ {v_in}).
    #    b) v_in is not needed for validity of others (removal always helps).
    #    So we just need to check v_out against (selected \ {v_in}).
    
    # Initialization: Greedy pack to get a good starting point.
    # Random order greedy.
    idx_order = np.random.permutation(N)
    selected = np.zeros(N, dtype=bool)
    
    # To speed up greedy, we can just iterate.
    # Since N is ~5000, O(N^2) greedy is ~25M ops, which is fast (<0.1s).
    for i in idx_order:
        if selected[i]:
            continue
        # Check if i is compatible with all selected
        # G[i, selected] <= max_ip_allowed
        if np.all(G[i, selected] <= max_ip_allowed):
            selected[i] = True
    
    best_selected = selected.copy()
    best_score = np.sum(selected)
    
    # Simulated Annealing Loop
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # We want to try many swaps/adds.
    # Precompute list of indices for faster random access? np.random.choice is fine.
    
    # Optimization:
    # Instead of checking G[i, selected] every time (which creates a boolean mask and does a reduction),
    # we can maintain a 'max_ip_with_selected' array for each candidate i.
    # max_ip_with_selected[i] = max(G[i, j] for j in selected)
    # When we add j: for all i not in selected, max_ip_with_selected[i] = max(max_ip_with_selected[i], G[i, j])
    # When we remove j: we would need to recompute max for all i, which is O(N^2) or O(N) if we store history.
    # Storing history is hard.
    # However, removals are rare in SA if we bias towards additions? No, we need balance.
    
    # Alternative: Just use the G matrix check. 
    # np.all(G[i, selected] <= max_ip_allowed) is O(N) vectorized.
    # 120s / 1e-6s per check (approx) = 120k iterations.
    # 120k iterations might not be enough for SA to converge well on 5000 items.
    
    # Let's try a hybrid: 
    # 1. Greedy to start.
    # 2. Local Search: Try to add any unselected vector that fits. Repeat until none fit.
    # 3. Then try to swap: Pick a selected vector s and an unselected vector u. 
    #    If removing s allows u to fit, and the "cost" is low? 
    #    Actually, simply: if we can find a pair (s, u) such that u fits in (selected \ {s}), we can swap.
    #    This increases size by 0 but might enable future additions.
    #    Better: If we can find u that fits in (selected \ {s1, s2, ...}), we can swap multiple?
    
    # Let's stick to a robust implementation of "Iterated Local Search" or just a long SA.
    # Given the time limit, a well-tuned SA with O(N) validity checks is viable.
    
    # To speed up validity check:
    # We can maintain a count of "bad neighbors" for each unselected node?
    # bad_count[i] = number of selected nodes j such that G[i, j] > max_ip_allowed.
    # If bad_count[i] == 0, i is valid to add.
    # When we add node j:
    #   For all i where G[i, j] > max_ip_allowed, bad_count[i] += 1.
    # When we remove node j:
    #   For all i where G[i, j] > max_ip_allowed, bad_count[i] -= 1.
    
    # This is O(N) per move!
    # Initialization of bad_count:
    # For each i, bad_count[i] = sum(selected[j] * (G[i, j] > max_ip_allowed))
    
    # Let's implement this.
    
    # 1. Greedy Start
    selected = np.zeros(N, dtype=bool)
    # Random greedy
    order = np.random.permutation(N)
    for i in order:
        if selected[i]:
            continue
        # Check G[i, selected]
        if np.all(G[i, selected] <= max_ip_allowed):
            selected[i] = True
            
    best_score = np.sum(selected)
    best_selected = selected.copy()
    
    # 2. Compute initial bad_counts
    # bad_counts[i] = number of selected j with G[i,j] > max_ip
    # We can compute this vectorized:
    # mask_bad = G > max_ip_allowed
    # bad_counts = mask_bad @ selected.astype(int)
    mask_bad = G > max_ip_allowed
    bad_counts = mask_bad.dot(selected.astype(np.int32))
    
    # Valid candidates are those with bad_counts == 0 and not selected.
    # Note: For selected i, bad_counts[i] includes itself? 
    # G[i,i] = 4 > 2. So if i is selected, mask_bad[i,i] is True.
    # selected[i] is 1. So bad_counts[i] includes 1 from itself.
    # We only care about bad_counts for UNSELECTED nodes.
    # For unselected i, bad_counts[i] is count of selected neighbors that conflict.
    
    # SA Loop
    T = 1.0
    T_min = 0.001
    cooling = 0.9995 # Slow cooling to allow more exploration
    
    # We will perform moves until time runs out.
    # Moves:
    # 1. Add: Pick i where bad_counts[i] == 0 and not selected.
    # 2. Remove: Pick i where selected.
    # 3. Swap: Pick selected i, unselected j. 
    #    Effectively: Remove i, Add j.
    #    Validity of j after removing i:
    #    bad_counts[j] - (1 if G[j,i] > max_ip else 0) == 0.
    
    # To make it efficient, we can maintain a list of "available" nodes (bad_counts == 0, not selected).
    # But maintaining this list dynamically is tricky.
    # Instead, we can just pick random nodes and check.
    # Since N=5000, picking random is cheap. Checking bad_counts[i] is O(1).
    
    # However, if we pick many invalid ones, we waste time.
    # Let's track the number of available nodes.
    # available_count = sum((~selected) & (bad_counts == 0))
    
    available_count = np.sum((~selected) & (bad_counts == 0))
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        r = np.random.random()
        
        if r < 0.5 and available_count > 0:
            # Try to Add
            # Pick a random node. If it's not available, we might need to retry or just skip.
            # To avoid wasting time on invalid picks when available_count is small, 
            # we can just pick random until we find a valid one (with a limit) or use a list.
            # For now, simple random pick.
            i = np.random.randint(N)
            if selected[i] or bad_counts[i] > 0:
                continue
            
            # Add i
            delta_score = 1
            # Update bad_counts for all j where G[j, i] > max_ip
            # indices_to_update = np.where(mask_bad[:, i])[0]
            # But mask_bad is N x N. Accessing column is O(N).
            # We can precompute the list of conflicting indices for each i?
            # conflict_lists[i] = np.where(mask_bad[i])[0]
            # This takes memory but speeds up update.
            # 5000 lists of avg length ~ N/2? 25M ints. 100MB. Okay.
            
            # Let's just do the vectorized update on the whole bad_counts array?
            # bad_counts += mask_bad[:, i].astype(int) * (1 - selected).astype(int) ? No, we only update unselected?
            # Actually, bad_counts is defined for all nodes.
            # If we add i, for any node j, if G[j,i] > max_ip, bad_counts[j] increases by 1.
            # We don't need to mask by selected[j] because bad_counts for selected nodes is not used for validity checks directly (we check bad_counts[i] for unselected i).
            # Wait, if j is selected, we don't care about its bad_count for adding.
            # But when we remove j, we need to decrement bad_counts for its neighbors.
            # So we must keep bad_counts accurate for ALL nodes.
            
            # Update:
            # mask_bad_col = mask_bad[:, i]
            # bad_counts[mask_bad_col] += 1
            
            # This is O(N) per move.
            # 120s / (5000 ops) = 24k moves.
            # 24k moves is small.
            # We need faster updates.
            
            # Optimization: Precompute conflict masks as sparse or lists?
            # Or just accept O(N) and hope for the best?
            # 24k moves might not be enough to beat greedy significantly.
            
            # Let's try to be smarter.
            # We can maintain the list of available nodes.
            # If we add i, we remove from available any node j that conflicts with i.
            # If we remove i, we add back to available any node j that only conflicted with i (i.e., bad_counts[j] becomes 0).
            
            # This requires knowing which nodes are "blocked only by i".
            # That's complex.
            
            # Let's stick to the simple O(N) update but optimize the Python loop overhead.
            # Using NumPy slicing is fast.
            
            # Update bad_counts
            # We need to add 1 to bad_counts[j] for all j where G[j, i] > max_ip.
            # conflict_mask = mask_bad[:, i]
            # bad_counts[conflict_mask] += 1
            
            # This is one vectorized op.
            # Let's see if we can do better.
            
            # Actually, let's just run the loop and see.
            
            # For now, let's implement the update.
            conflict_mask = mask_bad[:, i]
            bad_counts[conflict_mask] += 1
            
            selected[i] = True
            available_count -= 1 # i is no longer available
            # Also, any node j that was available but conflicts with i is no longer available.
            # We need to update available_count.
            # new_unavailable = conflict_mask & (~selected) & (bad_counts == 0) ???
            # No, bad_counts just increased.
            # Nodes that were available (bad_count 0) and conflict with i now have bad_count 1.
            # So they are no longer available.
            # count_new_unavail = np.sum(conflict_mask & (~selected) & (bad_counts == 1)) # Since they were 0, now 1.
            # But we just updated bad_counts.
            # So:
            # was_available = (~selected) & (bad_counts == 0) # BEFORE update?
            # We need to know who was available before.
            
            # This bookkeeping is getting complex and error-prone.
            # Let's simplify: Just recompute available_count every K moves? Or just use random sampling.
            # If we just use random sampling, we might waste time.
            
            # Alternative Strategy: 
            # Since N=5000 is small, we can just use the G matrix and do a fast check.
            # But SA needs many moves.
            
            # Let's try a different approach: 
            # **Random Restart Greedy with Local Search**
            # 1. Greedy pack.
            # 2. Try to add any vector that fits.
            # 3. If stuck, remove a random vector, try to add others.
            
            # This is simpler and might be faster.
            
            # Let's implement "Greedy + Random Restart".
            # We have 120s.
            # One greedy run is ~0.1s.
            # We can do 1000 restarts.
            # Each restart:
            #   - Greedy pack (0.1s)
            #   - Local search: Try to add any unselected vector. (O(N^2) worst case, but usually fast if we maintain a list of candidates)
            
            # Let's implement Local Search properly.
            
            # Break out of current SA attempt if it's too slow/complex.
            # Let's switch to **Iterated Local Search (ILS)** with Greedy Initialization.
            
            # ILS:
            # 1. x = Greedy()
            # 2. x = LocalSearch(x)
            # 3. x' = Perturb(x)
            # 4. x = LocalSearch(x')
            # 5. if x' better than x, accept.
            
            # LocalSearch:
            #   Maintain list of "open" vectors (unselected and compatible with all selected).
            #   While open list not empty:
            #       Pick one, add to selected.
            #       Update open list: remove any that conflict with new vector.
            #       Add any that become open? No, adding a vector only creates conflicts, doesn't remove them.
            #       So open list only shrinks.
            #   This is O(N^2) per LocalSearch call?
            #   Initial open list construction: O(N^2).
            #   Each addition: O(N) to update open list.
            #   Total O(N^2).
            #   With N=5000, 25M ops. ~0.05s.
            #   We can do 1000 restarts in 50s.
            
            # This is a solid strategy.
            
            # Let's implement this instead of SA.
            pass # Break out of SA logic

    # Implementation of ILS
    
    def _greedy_pack(candidates, G, max_ip, order):
        selected = np.zeros(len(candidates), dtype=bool)
        for i in order:
            if selected[i]:
                continue
            if np.all(G[i, selected] <= max_ip):
                selected[i] = True
        return selected

    def _local_search(candidates, G, max_ip, selected):
        # Find all unselected vectors that are compatible with all selected
        N = len(candidates)
        # open_mask[i] = True if i is unselected and compatible with all selected
        # G[:, selected] gives matrix of IPs between all i and selected.
        # max_ip_in_selected = np.max(G[:, selected], axis=1)
        # open_mask = (~selected) & (max_ip_in_selected <= max_ip)
        
        # To be efficient, we can compute this once.
        # But as we add vectors, we need to update.
        
        # Let's use a list of open indices.
        # Initial open list:
        # We can compute max_ip_with_selected for all i.
        # max_ip_ws = np.max(G[:, selected], axis=1) if any selected else -inf
        # But np.max on empty slice fails.
        
        if np.sum(selected) == 0:
            open_indices = np.where(~selected)[0]
        else:
            # G[:, selected] is N x K.
            # We want max along axis 1.
            # If K is large, this is O(NK).
            max_ips = np.max(G[:, selected], axis=1)
            open_indices = np.where((~selected) & (max_ips <= max_ip))[0]
        
        # Now, while open_indices is not empty:
        # Pick one (e.g., random or first), add it.
        # Update open_indices: remove any j in open_indices where G[j, new_i] > max_ip.
        
        while len(open_indices) > 0:
            # Pick a random one to diversify? Or just pop.
            # Popping from end is fast.
            idx = open_indices[-1]
            open_indices = open_indices[:-1]
            
            # Add idx
            selected[idx] = True
            
            # Update open_indices
            # Any j in open_indices with G[j, idx] > max_ip must be removed.
            # conflict_mask = G[open_indices, idx] > max_ip
            # open_indices = open_indices[~conflict_mask]
            
            if len(open_indices) > 0:
                conf = G[open_indices, idx] > max_ip
                if np.any(conf):
                    open_indices = open_indices[~conf]
                    
        return selected

    # Main Loop
    best_selected = None
    best_score = 0
    
    # Initial greedy
    order = np.random.permutation(N)
    sel = _greedy_pack(candidates, G, max_ip_allowed, order)
    sel = _local_search(candidates, G, max_ip_allowed, sel)
    score = np.sum(sel)
    if score > best_score:
        best_score = score
        best_selected = sel.copy()
        
    # ILS Loop
    while time.time() < deadline:
        # Perturb: Remove K random vectors from current best
        # K can be 1, 2, or 5.
        K = np.random.choice([1, 2, 5])
        if best_score < K:
            continue
            
        current = best_selected.copy()
        # Pick K random selected indices to remove
        selected_indices = np.where(current)[0]
        if len(selected_indices) == 0:
            continue
        remove_idx = np.random.choice(selected_indices, size=K, replace=False)
        current[remove_idx] = False
        
        # Local Search
        # Note: _local_search assumes 'selected' is the starting point.
        # It finds open vectors and adds them.
        # It does NOT remove vectors.
        # So it will just try to fill the gaps.
        new_sel = _local_search(candidates, G, max_ip_allowed, current)
        new_score = np.sum(new_sel)
        
        if new_score > best_score:
            best_score = new_score
            best_selected = new_sel.copy()
            
    # Return best found
    if best_selected is not None and best_score > 0:
        return candidates[best_selected]
    else:
        return _generate_seed_vectors()

# EVOLVE-BLOCK-END
