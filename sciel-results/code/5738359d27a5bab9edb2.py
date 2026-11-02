import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_init():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        # Shuffle to break symmetry
        order = rng.permutation(N4)
        
        for v in order:
            if not alive[v]:
                continue
            # Check if v is still valid (no neighbors in set)
            # Since we remove neighbors when adding, alive[v] being True means no neighbor was added?
            # No, alive[v] becomes False if v is added OR if a neighbor is added.
            # So if alive[v] is True, v is not in set and no neighbor is in set.
            in_set[v] = True
            nbrs = neighbors[v]
            alive[v] = False
            alive[nbrs] = False
            
        return in_set

    def local_search_improve(in_set_init, max_time=50.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        # count_conflicts[v] = number of neighbors of v in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        sa_deadline = time.time() + max_time
        T = 1.0
        min_T = 0.1
        cooling = 0.999
        
        while time.time() < sa_deadline:
            # 1. Add all 0-conflict nodes
            improved = True
            while improved:
                improved = False
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0: break
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                if len(zero_cands) > 0:
                    for v in zero_cands:
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            # 2. Try Remove 1, Add k moves
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) == 0: break
            
            # Sample a few nodes to try removing to save time
            num_tries = min(100, len(in_set_indices))
            try_indices = in_set_indices[rng.choice(len(in_set_indices), size=num_tries, replace=False)]
            
            for u in try_indices:
                if time.time() > sa_deadline: break
                
                # Temporarily remove u
                in_set[u] = False
                current_size -= 1
                count_conflicts[neighbors[u]] -= 1
                
                # Find candidates v that are neighbors of u and now have 0 conflicts
                # v must be not in set (was not, since u was in set and v~u implies v not in set? 
                # No, v could be in set? No, if v in set and v~u, they would conflict. 
                # So all neighbors of u are NOT in set.)
                nbrs_u = neighbors[u]
                # Check which nbrs_u now have 0 conflicts
                # count_conflicts[v] == 0
                valid_v = nbrs_u[count_conflicts[nbrs_u] == 0]
                
                if len(valid_v) == 0:
                    # Restore u
                    in_set[u] = True
                    current_size += 1
                    count_conflicts[neighbors[u]] += 1
                    continue
                
                # We can add a subset of valid_v.
                # valid_v are all neighbors of u.
                # Do they conflict with each other?
                # We need to find a Max Independent Set in the subgraph induced by valid_v.
                # Since valid_v is small, we can just greedily add.
                added = []
                # Sort by degree in the subgraph? Or just random?
                # Let's just greedily add:
                for v in valid_v:
                    # Check if v conflicts with any already added in this step
                    # v conflicts with w if conflicts[v, w]
                    # We can check against `added`
                    ok = True
                    for w in added:
                        if conflicts[v, w]:
                            ok = False
                            break
                    if ok:
                        added.append(v)
                
                if len(added) > 1:
                    # Gain!
                    for v in added:
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                    # Do not restore u
                elif len(added) == 1:
                    # Neutral move, might help escape
                    # Accept with probability
                    if rng.random() < 0.1:
                        v = added[0]
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                    else:
                        # Restore u
                        in_set[u] = True
                        current_size += 1
                        count_conflicts[neighbors[u]] += 1
                else:
                    # Restore u
                    in_set[u] = True
                    current_size += 1
                    count_conflicts[neighbors[u]] += 1

            T *= cooling
            if T < min_T: T = min_T

        return local_best_in_set, local_best_size

    # Run multiple restarts
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    # Initial greedy
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Improve
    improved_set, improved_size = local_search_improve(in_set, max_time=100.0)
    if improved_size > best_size:
        best_size = improved_size
        best_in_set = improved_set.copy()

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
