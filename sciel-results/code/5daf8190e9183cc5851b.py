"""Kissing number in dimension 11 - Improved MIS search."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    # Conflict matrix
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    
    rng = np.random.default_rng(42)

    def randomized_greedy():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            # Random selection among min-degree candidates
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search(in_set):
        """
        Improves the set by:
        1. Adding any vertex with 0 conflicts.
        2. Swapping: If a vertex v has 1 conflict u, remove u and add v.
           (This maintains size but changes structure, potentially enabling future adds)
        3. If stuck, try removing a random vertex from set to free up space? 
           (Handled by restart/perturbation in main loop)
        """
        in_set = in_set.copy()
        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1
        
        changed = True
        # Limit iterations to avoid infinite loops if no progress
        max_iter = 1000
        iter_count = 0
        
        while changed and iter_count < max_iter and time.time() < deadline - 10:
            changed = False
            iter_count += 1
            
            # 1. Try to add vertices with 0 conflicts
            candidates_add = np.where((cc == 0) & (~in_set))[0]
            for v in candidates_add:
                if cc[v] == 0 and not in_set[v]:
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    changed = True
            
            # 2. Try swaps: v not in set, cc[v] == 1
            candidates_swap = np.where((cc == 1) & (~in_set))[0]
            for v in candidates_swap:
                if cc[v] == 1 and not in_set[v]:
                    # Find the single conflict
                    u = np.where(conflicts[v] & in_set)[0]
                    if len(u) == 1:
                        u = u[0]
                        # Remove u, add v
                        in_set[u] = False
                        cc[neighbors[u]] -= 1
                        in_set[v] = True
                        cc[neighbors[v]] += 1
                        changed = True

        return in_set

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # Run multiple restarts
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = randomized_greedy()
        in_set = local_search(in_set)
        cur_size = int(in_set.sum())
        
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()
            
        # Perturbation: Remove a few random elements from best set and re-optimize
        if n_restarts > 1 and time.time() < deadline - 20:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 5:
                to_remove = rng.choice(indices, size=5, replace=False)
                perturbed[to_remove] = False
                perturbed = local_search(perturbed)
                psize = int(perturbed.sum())
                if psize > best_size:
                    best_size = psize
                    best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]

    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
