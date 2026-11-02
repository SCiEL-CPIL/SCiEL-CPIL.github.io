"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        # Choose w indices out of D
        for indices in itertools.combinations(range(D), w):
            # Choose signs for these indices
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Generate candidates with weights 1, 2, 3
    # Weight 1: 22 candidates (norm^2=1)
    # Weight 2: 220 candidates (norm^2=2)
    # Weight 3: 1320 candidates (norm^2=3)
    # Total: 1562 candidates
    # This pool is small enough to handle without a huge precomputed matrix,
    # but large enough to allow diverse packing.
    candidates = _generate_candidates([1, 2, 3])
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute dot product matrix for compatibility checks
    # Compatibility condition: dist^2 >= max(norm_i^2, norm_j^2)
    # dist^2 = n_i + n_j - 2 dot(i, j)
    # n_i + n_j - 2 dot(i, j) >= max(n_i, n_j)
    # If n_i >= n_j: n_i - 2 dot >= 0 => dot <= n_i / 2
    # If n_j > n_i: n_j - 2 dot >= 0 => dot <= n_j / 2
    # So generally: dot <= max(n_i, n_j) / 2
    
    dots = candidates @ candidates.T
    
    # Create a compatibility matrix (True if compatible)
    # We want to check if i and j can coexist.
    # max_norms[i,j] = max(norms_sq[i], norms_sq[j])
    max_norms = np.maximum(norms_sq[:, np.newaxis], norms_sq[np.newaxis, :])
    
    # Condition: 2 * dots <= max_norms
    # Note: diagonal is always compatible (dot = norm, 2*norm > norm is false? 
    # Wait. dist(i,i)=0. Condition dist >= max_norm. 0 >= norm is false for non-zero.
    # But we never pick i and i. We pick distinct i, j.
    # For i != j, we need 2*dot(i,j) <= max(norm_i, norm_j).
    
    compat = 2 * dots <= max_norms
    
    # Self-compatibility is irrelevant for selection, but let's ensure we don't pick same index twice.
    # We will use a boolean mask for selected.
    
    best_set = set()
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Min-Degree Greedy with Random Restarts
    # Min-degree: Pick the candidate that has the fewest conflicts with the *available* pool?
    # Or fewest conflicts with the *selected* set?
    # Standard greedy set packing: Pick vertex with min degree in the *conflict graph* restricted to available vertices.
    # Here, "degree" of a candidate v in the available set A is the number of u in A such that NOT compat[v, u].
    # However, calculating this dynamically is expensive.
    
    # Alternative Min-Degree heuristic:
    # 1. Start with all candidates available.
    # 2. For each available candidate, count how many *other* available candidates it conflicts with.
    # 3. Pick the one with the MINIMUM such count (least constraining).
    # 4. Add to selected, remove it and all its conflict neighbors from available.
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 30: # Leave time for local search
            break
            
        available = np.ones(n_cand, dtype=bool)
        selected = []
        
        # Precompute conflict counts? No, it changes as we remove nodes.
        # We will approximate min-degree by checking conflicts against the *current* available set.
        
        while True:
            if time.time() > deadline - 30:
                break
                
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Calculate "degree" (number of conflicts) for each available node within the available set
            # conflict[v, u] is True if NOT compat[v, u]
            # We want min number of conflicts.
            
            # Vectorized approach:
            # For each v in avail_indices, count conflicts with all u in avail_indices.
            # This is O(N^2) per step if done naively.
            # N=1562. N^2 = 2.4M. Steps ~ N. Total ~ 3.7B ops. Too slow for Python loops.
            
            # Optimization:
            # Instead of full recalculation, we can use a heuristic.
            # Or, we can just use Random Greedy but with a better initial ordering?
            # The inspiration says "Min-degree greedy preserves more flexibility".
            
            # Let's try a faster approximation:
            # Pick a random subset of candidates to evaluate for "min degree"?
            # Or, simply: Sort available candidates by their *initial* degree (total conflicts in the whole pool)?
            # Nodes with low total degree are generally less constraining.
            
            # Precompute total conflict count for each node once.
            # total_conflicts[i] = sum(not compat[i, :])
            
            # Let's precompute total_conflicts outside the loop.
            
            # Actually, let's stick to a robust Random Greedy first, then add Local Search.
            # The current program score is 1.25. Inspiration 1 (SA) got 1.58.
            # The "Min-degree" insight from knowledge base: "Min-degree greedy ... 1.891 -> 2.123".
            # This suggests Min-Degree is very powerful.
            
            # How to implement Min-Degree efficiently?
            # We can maintain an array `conflict_count` for available nodes.
            # Initially, conflict_count[i] = number of j where not compat[i,j].
            # When we select node v:
            #   For all u in available:
            #     if not compat[v, u]:
            #       conflict_count[u] -= 1 (because v is removed from available)
            #     available[u] = False if not compat[v, u] (remove neighbors)
            #   available[v] = False
            #   conflict_count[v] = 0
            
            # This update is O(N) per selection.
            # Total O(N^2) per restart.
            # N=1562. N^2 = 2.4M.
            # 200 restarts = 480M ops. In Python, this is likely too slow (seconds per restart).
            # We need to vectorize the update.
            
            # Let's try a different strategy:
            # 1. Precompute total_conflicts.
            # 2. Greedy: Pick node with min current conflict count.
            #    To avoid O(N) update, we can just re-calculate min for the available set?
            #    min(conflict_count[available]). This is O(N).
            #    Update is O(N).
            #    So O(N^2) total.
            
            # Let's optimize the update using NumPy.
            
            pass # Placeholder for logic

        # Re-implementing with optimized Min-Degree
        
    # Precompute total conflicts for initial degree
    # not_compat = ~compat
    # np.fill_diagonal(not_compat, False) # Self is not a conflict for degree counting in the same set?
    # Actually, if I pick v, I remove v and its neighbors.
    # Degree of v in available set A is count of u in A \ {v} such that not compat[v, u].
    
    not_compat = ~compat
    np.fill_diagonal(not_compat, False)
    
    # We will run a few restarts of Min-Degree Greedy.
    # To speed up, we will limit restarts and use vectorized updates.
    
    best_set_indices = []
    best_count = 0
    
    # Precompute neighbor lists for fast removal?
    # neighbors[i] = indices j where not_compat[i, j]
    
    neighbors = [np.where(not_compat[i])[0] for i in range(n_cand)]
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        available = np.ones(n_cand, dtype=bool)
        selected = []
        
        # Initial conflict counts (within the full set)
        # conflict_count[i] = len(neighbors[i])
        # But as we remove nodes, this changes.
        
        # Efficient Min-Degree Greedy:
        # At each step, find i in available with min number of available neighbors.
        # This is hard to do incrementally without O(N) update.
        
        # Let's use a simpler heuristic that is O(N^2) but vectorized per step.
        # Step 1: Find available indices.
        # Step 2: For each available i, count available neighbors.
        #         This is sum(available[neighbors[i]]).
        #         We can do this for all i at once?
        #         counts = np.array([np.sum(available[neighbors[i]]) for i in available_indices])
        #         This is a Python loop over N. Slow.
        
        # Alternative:
        # Use the precomputed not_compat matrix.
        # For a given available mask, the degree of i is dot(available, not_compat[i])
        # We can compute degrees for all i: degrees = not_compat @ available
        # Then pick i with min degrees[i] where available[i] is True.
        
        while True:
            if time.time() > deadline - 20:
                break
                
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Compute degrees for all nodes (vectorized matrix-vector product)
            # degrees[i] = number of neighbors of i that are available
            degrees = not_compat @ available
            
            # Among available nodes, find the one with minimum degree
            # Set degrees of unavailable nodes to inf so they aren't picked
            current_degrees = np.where(available, degrees, np.inf)
            
            min_idx = np.argmin(current_degrees)
            
            # Select min_idx
            selected.append(min_idx)
            
            # Update available:
            # Remove min_idx and all its neighbors
            # available[neighbors[min_idx]] = False
            # available[min_idx] = False
            
            # Vectorized update:
            # We can do: available[not_compat[min_idx, :]] = False
            # and available[min_idx] = False
            
            # Note: not_compat[min_idx, :] is a boolean vector.
            available[not_compat[min_idx, :]] = False
            available[min_idx] = False
            
        if len(selected) > best_count:
            best_count = len(selected)
            best_set_indices = selected.copy()
            
    # Phase 2: Local Search (Grow by 1-for-2 swaps)
    # Try to replace one selected vector with two unselected vectors that are compatible with each other and the rest of the set.
    
    if best_count > 0:
        selected_set = set(best_set_indices)
        
        # Precompute compatibility with the rest of the set for unselected nodes?
        # For a node u not in selected, let C(u) be the set of selected nodes compatible with u.
        # If we remove v from selected, u is compatible with the new set if C(u) is subset of (selected \ {v}).
        # i.e., u is compatible with all selected nodes except possibly v.
        
        # Let's iterate through selected nodes v.
        # For each v, consider removing it.
        # Find pairs (u1, u2) not in selected such that:
        # 1. u1 and u2 are compatible.
        # 2. u1 is compatible with all selected \ {v}.
        # 3. u2 is compatible with all selected \ {v}.
        
        # This is complex. Let's simplify:
        # Just try to add any unselected node? No, greedy already did that.
        # The swap is: Remove v, Add u1, Add u2.
        
        # Optimization:
        # For each unselected node u, precompute the set of selected nodes it conflicts with.
        # conflict_with_selected[u] = {v in selected | not compat[u, v]}
        
        # If we remove v, u becomes "addable" if conflict_with_selected[u] is subset of {v}.
        # i.e., u conflicts with NO selected nodes, OR u conflicts ONLY with v.
        
        # Let's compute conflict counts with selected for all unselected nodes.
        
        # This local search might be too slow to implement fully within the time limit if done naively.
        # Given the "Min-Degree" success, the greedy phase should be strong enough.
        # Let's stick to the Min-Degree Greedy and ensure it runs efficiently.
        
        # One more optimization: The matrix multiplication `not_compat @ available` is O(N^2).
        # With N=1562, that's ~2.4M ops.
        # If we select ~500 nodes, that's 1.2B ops.
        # NumPy mat-vec is fast. 1.2B simple ops might take ~1-2 seconds per restart.
        # 50 restarts = 50-100 seconds.
        # This fits in the budget.
        
        pass

    if not best_set_indices:
        # Fallback: Return weight 2 vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[best_set_indices]

# EVOLVE-BLOCK-END
