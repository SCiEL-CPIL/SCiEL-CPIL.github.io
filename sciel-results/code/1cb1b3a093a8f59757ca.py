import numpy as np
import itertools
import time
import random
from collections import deque

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570
    d = 11

    # Fallback: D11 minimal vectors (Norm 2) - 220 vectors
    # Type: two entries are +/-1, rest 0
    fallback_list = []
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                fallback_list.append(v)
    fallback_points = np.array(fallback_list)

    # Generate Norm 4 candidates
    # Type 1: One entry +/-2 (2*11 = 22 vectors)
    # Type 2: Four entries +/-1 (C(11,4) * 16 = 330 * 16 = 5280 vectors)
    candidates_list = []

    # Type 1: +/-2
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates_list.append(v)

    # Type 2: +/-1 on 4 coords
    for idxs in itertools.combinations(range(d), 4):
        i, j, k, l = idxs
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates_list.append(v)

    candidates = np.array(candidates_list)
    n_cands = len(candidates)

    # We want to find a maximum subset such that pairwise dot product <= 2.
    # This is equivalent to Maximum Independent Set on the graph where an edge exists if dot > 2.
    # Or Maximum Clique on the graph where an edge exists if dot <= 2.

    # Let's compute the compatibility matrix.
    # compatible[i, j] = True if dot(candidates[i], candidates[j]) <= 2
    # Note: dot is integer.
    # We can use matrix multiplication.
    dot_matrix = candidates @ candidates.T

    # Conflict if dot > 2.
    # Compatible if dot <= 2.
    compatible = dot_matrix <= 2
    # Diagonal is 4, so compatible[i,i] is False. Good.

    # Now we have a boolean matrix `compatible` of size (n_cands, n_cands).
    # We want to find the largest set of indices such that all pairs are compatible.

    # Heuristic: Greedy with randomization and local search.
    # We can also use a specific algorithm for Maximum Clique / Independent Set.
    # Bron-Kerbosch with pivot?
    # But n_cands is 5302.
    # Let's try a greedy approach first, as it's fast.
    # Then maybe improve with a few swaps.

    def greedy_is(compatible, n, order):
        selected = []
        selected_set = set()
        # For each node in order, check if it's compatible with all selected nodes.
        # We can maintain a list of selected indices and check.
        # To speed up, we can maintain a bitmask? No, n is 5302, too large for standard int bitmask unless we use Python big ints or numpy bool arrays.
        # Using numpy bool array for selected nodes might be efficient if we slice.
        # But checking `compatible[idx, selected]` creates a view.
        
        # Alternative: maintain a "forbidden" mask.
        # When we select a node, all its neighbors (conflicts) become forbidden.
        # But we want an Independent Set.
        # If we select node i, we cannot select any j such that not compatible[i, j].
        # Let's maintain a boolean array `available` of size n.
        # Initially all True.
        # When we select i, we set available[j] = False for all j where not compatible[i, j].
        # This is O(n) per selection. If we select ~200 nodes, that's 200 * 5302 ~ 1M ops. Very fast.
        
        available = np.ones(n, dtype=bool)
        for idx in order:
            if not available[idx]:
                continue
            # Select idx
            selected.append(idx)
            # Update available: remove all conflicts of idx
            # Conflicts are where compatible[idx, :] is False
            # So available[:] = available[:] & compatible[idx, :]
            available[:] = available[:] & compatible[idx, :]
            # Set available[idx] to False explicitly to avoid re-selection (though it's already handled by the loop logic if we break, but we don't break, we continue)
            # Actually, compatible[idx, idx] is False, so available[idx] becomes False.
        return selected

    def greedy_is_with_swap(compatible, n, order, max_swaps=100):
        selected = greedy_is(compatible, n, order)
        selected_set = set(selected)
        # Try to add any node? No, greedy already did that.
        # Try swaps: remove one, add two?
        # This is complex. Let's stick to multiple random orders first.
        return selected

    best_size = 0
    best_indices = []

    # Strategy 1: Greedy by degree (number of compatible neighbors)
    degrees = compatible.sum(axis=1)
    # Higher degree means more compatible neighbors, so better to pick early?
    # Or lower degree (more constrained) should be picked earlier?
    # Usually for Independent Set, picking nodes with low degree in the *conflict* graph (high degree in compatible) is good?
    # Let's try sorting by degree descending (most compatible first).
    order_deg = np.argsort(-degrees)
    sel_deg = greedy_is(compatible, n_cands, order_deg)
    if len(sel_deg) > best_size:
        best_size = len(sel_deg)
        best_indices = sel_deg

    # Strategy 2: Random orders
    # We have time. Let's run several random restarts.
    n_restarts = 50
    # Estimate time per restart.
    # Greedy is O(n * size). Size ~ 200. n = 5302. ~ 1M ops.
    # 50 restarts ~ 50M ops. Should be < 1s in Python/Numpy?
    # The vectorized update `available[:] = available[:] & compatible[idx, :]` is fast.
    
    # Let's use a deadline check inside the loop.
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        order_rand = np.random.permutation(n_cands)
        sel_rand = greedy_is(compatible, n_cands, order_rand)
        if len(sel_rand) > best_size:
            best_size = len(sel_rand)
            best_indices = sel_rand

    # Strategy 3: Local Search / Improvement
    # Try to improve the best set found.
    # Can we swap?
    # Let's try a simple heuristic: 
    # Take the best set. 
    # Look at nodes not in the set. 
    # If a node is compatible with most of the set, maybe it can be added if we remove a few conflicting ones?
    # This is getting complicated.
    
    # Alternative: Use a specific MIS solver if available? No, only stdlib/NumPy/SciPy.
    # SciPy doesn't have a general MIS solver.
    
    # Let's try "Greedy with lookahead" or "Coloring based bound"?
    # For now, let's see if we can beat 220 with the simple greedy.
    
    if best_size > len(fallback_points):
        # Validate the result
        best_points = candidates[best_indices]
        # Check constraints
        # 1. No zero vector (all have norm 4, so ok)
        # 2. min dist sq >= max norm sq
        # Norms are all 4.
        # Dist sq = 8 - 2 dot.
        # We ensured dot <= 2, so dist sq >= 8 - 4 = 4.
        # So min dist sq >= 4 = max norm sq.
        # Valid!
        return np.around(best_points).astype(np.int64)
    else:
        return np.around(fallback_points).astype(np.int64)

# Helper to ensure no crash if time runs out during validation (though validation is fast)
# The main risk is the greedy loop taking too long if n_cands was larger, but it's fixed.
