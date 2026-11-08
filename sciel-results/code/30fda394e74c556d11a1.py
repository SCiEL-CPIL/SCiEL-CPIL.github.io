"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Strategy: Mixed-weight candidate pool (w=2,3,4) with vectorized greedy independent set.
    # Conflict condition: for vectors u,v with norms ||u||,||v||, we need ||u-v|| >= max(||u||,||v||).
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v >= max(||u||^2, ||v||^2)
    # For w=2: ||u||^2=2, need 2+2-2uv >= 2 => uv <= 1
    # For w=3: ||u||^2=3, need 3+3-2uv >= 3 => uv <= 1.5 => uv <= 1 (integer)
    # For w=4: ||u||^2=4, need 4+4-2uv >= 4 => uv <= 2
    # Mixed: w_a, w_b with norms^2 = a, b: need a+b-2uv >= max(a,b) => uv <= (a+b-max(a,b))/2 = min(a,b)/2
    # So: w=2,w=2: uv<=1; w=2,w=3: uv<=1; w=2,w=4: uv<=1; w=3,w=3: uv<=1; w=3,w=4: uv<=1; w=4,w=4: uv<=2

    # Generate all candidates
    candidates = []
    # Weight 2
    for support in combinations(range(D), 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    # Weight 3
    for support in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    # Weight 4
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)

    # Compute norms squared
    norms_sq = np.sum(cand_arr ** 2, axis=1).astype(np.int64)

    # Compute Gram matrix
    G = (cand_arr.astype(np.int32) @ cand_arr.T.astype(np.int32))

    # Conflict matrix: conflict[i,j] = True if NOT compatible
    # Compatible if G[i,j] <= min(norms_sq[i], norms_sq[j]) // 2
    # Since all norms are 2,3,4: min//2 gives 1,1,1,1,1,2
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    threshold = min_norms // 2
    conflict = G > threshold  # True means conflict (not allowed together)

    # Diagonal: self is always "conflict" with itself for our purposes (can't pick twice)
    np.fill_diagonal(conflict, True)

    # Precompute conflict as boolean matrix for fast vectorized operations
    # conflict[i] is a boolean array of length n_cands

    best_indices = []
    best_score = 0
    rng = np.random.default_rng(42)

    def vectorized_greedy_random_order():
        """
        Vectorized greedy: pick random order, for each candidate check if compatible
        with all already-selected using vectorized dot product check.
        """
        if time.time() > deadline - 3:
            return []

        order = rng.permutation(n_cands)
        selected = []
        selected_mask = np.zeros(n_cands, dtype=bool)

        for idx in order:
            if time.time() > deadline - 2:
                break

            # Check if idx is compatible with all selected
            if len(selected) == 0:
                selected.append(idx)
                selected_mask[idx] = True
                continue

            # Vectorized check: conflict[idx, selected] should all be False
            sel_arr = np.array(selected, dtype=np.int64)
            if not np.any(conflict[idx, sel_arr]):
                selected.append(idx)
                selected_mask[idx] = True

        return selected

    def vectorized_greedy_low_degree():
        """
        Greedy that prioritizes candidates with fewer conflicts among remaining.
        Uses vectorized degree computation.
        """
        if time.time() > deadline - 3:
            return []

        active = np.ones(n_cands, dtype=bool)
        selected = []
        remaining_conflict_count = np.sum(conflict, axis=1).astype(np.int32)

        while np.any(active):
            if time.time() > deadline - 2:
                break

            # Find active nodes
            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                break

            # Among active, pick one with minimum remaining conflicts (with other active)
            # Compute degree within active subgraph
            active_deg = np.sum(conflict[np.ix_(active_idx, active_idx)], axis=1)
            # Add small random noise to break ties
            noise = rng.random(len(active_idx)) * 0.1
            scores = active_deg + noise
            best_local = np.argmin(scores)
            node = active_idx[best_local]

            selected.append(node)
            active[node] = False

            # Deactivate all neighbors of node that are active
            neighbors_of_node = np.where(conflict[node] & active)[0]
            active[neighbors_of_node] = False

        return selected

    def local_search_add(initial_selected, max_passes=10):
        """
        Try to add more vectors. For each unselected candidate, check if compatible
        with current set. Add if so. Repeat.
        """
        if time.time() > deadline - 5:
            return initial_selected

        current = list(initial_selected)
        current_set_mask = np.zeros(n_cands, dtype=bool)
        for i in current:
            current_set_mask[i] = True

        for _pass in range(max_passes):
            if time.time() > deadline - 2:
                break

            added_any = False
            # Check all unselected candidates
            unselected = np.where(~current_set_mask)[0]
            if len(unselected) == 0:
                break

            for idx in unselected:
                if time.time() > deadline - 2:
                    break
                # Check compatibility with current set
                if len(current) == 0:
                    current.append(idx)
                    current_set_mask[idx] = True
                    added_any = True
                else:
                    sel_arr = np.array(current, dtype=np.int64)
                    if not np.any(conflict[idx, sel_arr]):
                        current.append(idx)
                        current_set_mask[idx] = True
                        added_any = True

            if not added_any:
                break

        return current

    # Run multiple strategies
    # Strategy 1: Random order greedy (fast, vectorized)
    for r in range(5):
        if time.time() > deadline - 30:
            break
        result = vectorized_greedy_random_order()
        if result:
            improved = local_search_add(result)
            if len(improved) > best_score:
                best_score = len(improved)
                best_indices = improved

    # Strategy 2: Low-degree greedy
    for r in range(3):
        if time.time() > deadline - 20:
            break
        result = vectorized_greedy_low_degree()
        if result:
            improved = local_search_add(result)
            if len(improved) > best_score:
                best_score = len(improved)
                best_indices = improved

    # Strategy 3: More random restarts with local search
    for r in range(20):
        if time.time() > deadline - 5:
            break
        result = vectorized_greedy_random_order()
        if result:
            improved = local_search_add(result, max_passes=3)
            if len(improved) > best_score:
                best_score = len(improved)
                best_indices = improved

    # Fallback: weight-2 only construction
    if best_score == 0:
        base_vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        base_vectors.append(v)
        best_indices = list(range(len(base_vectors)))
        # Need to map back to cand_arr indices
        # Actually just return the base vectors directly
        return np.array(base_vectors, dtype=np.int64)

    return cand_arr[best_indices].astype(np.int64)

# EVOLVE-BLOCK-END
