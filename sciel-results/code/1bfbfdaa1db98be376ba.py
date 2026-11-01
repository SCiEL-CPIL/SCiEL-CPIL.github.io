"""Kissing number in dimension 11 - Improved SA and Local Search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates: norm squared 4
    # Type A: weight 4, entries in {-1, 1}
    # Type B: weight 1, entries in {-2, 2}
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix
    # Conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to list of lists for faster iteration in SA
    # Or keep as numpy array for vectorized ops
    # For SA, we need fast neighbor checks.
    # neighbors[i] = list of j such that conflicts[i,j] is True
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init():
        """Greedy min-degree initialization."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def simulate_annealing(initial_solution, time_limit):
        """Simulated Annealing to improve independent set."""
        x = initial_solution.copy()
        # conflict_count[i] = number of neighbors of i in x
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(x)[0]:
            conflict_count[neighbors[i]] += 1
        
        current_size = x.sum()
        best_x = x.copy()
        best_size = current_size
        
        T = 1.0
        cooling = 0.999
        ls_deadline = time.time() + time_limit
        
        while time.time() < ls_deadline:
            # Pick a random vertex to flip
            i = rng.integers(N)
            
            if x[i]:
                # Remove i
                # Delta Size = -1
                # Delta Conflict Penalty?
                # We use objective: Size - Penalty * Conflicts
                # But we want to maintain independence.
                # Standard SA for MIS:
                # Move: Flip i.
                # If i in x:
                #   Remove i. Size decreases by 1.
                #   Conflicts for neighbors of i decrease.
                #   If any neighbor had conflict_count == 1 (only conflict was i), it becomes 0.
                #   This might allow adding them later.
                # If i not in x:
                #   Add i. Size increases by 1.
                #   But if conflict_count[i] > 0, we create conflicts.
                #   We allow this temporarily.
                
                # Let's use a simpler objective:
                # Score = Size - Lambda * TotalConflicts
                # TotalConflicts = sum(conflict_count) / 2
                
                # Calculate delta
                if x[i]:
                    # Remove i
                    delta_size = -1
                    # Neighbors of i lose one conflict
                    nbrs = neighbors[i]
                    # Count how many neighbors had conflict_count == 1
                    # Those will become 0 (good)
                    # Others remain > 0
                    # Total conflicts change:
                    # Each neighbor j of i had conflict_count[j] >= 1.
                    # After removal, conflict_count[j] decreases by 1.
                    # The term in objective for j is -Lambda * conflict_count[j].
                    # So delta penalty = -Lambda * sum(1 for j in nbrs) = -Lambda * degree(i)
                    # Wait, the objective is defined on the state.
                    # Let's just calculate the new score directly for small N? No, N is large.
                    
                    # Let's use a simpler heuristic:
                    # Only accept moves that maintain independence or improve size?
                    # No, SA needs to escape local minima.
                    
                    # Let's stick to the "Penalty" method.
                    # Objective: F = sum(x) - P * sum_{(i,j) in E} x_i x_j
                    # sum_{(i,j) in E} x_i x_j = 0.5 * sum_i x_i * conflict_count[i]
                    
                    # Current F:
                    # F_curr = current_size - P * 0.5 * sum(x * conflict_count)
                    
                    # This is getting complicated to track deltas efficiently.
                    # Let's use a different approach: 
                    # Local Search with 2-swaps is more direct.
                    pass
            else:
                pass
            
            # This SA implementation is getting complex.
            # Let's go back to a robust Local Search with 2-swaps.
            break 

    def local_search_2swap(in_set, max_time=5.0):
        """Local search with 1-add, 1-1 swap, and 1-2 swap (remove 1, add 2)."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < ls_deadline:
            improved = False
            
            # 1. Try to add a vertex with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    stuck = 0
                    continue
            
            # 2. Try 1-1 Swap
            if not improved and stuck > 10:
                cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                if len(cands_v) > 0:
                    v = cands_v[rng.integers(len(cands_v))]
                    nbrs_v_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_v_in_set) == 1:
                        u = nbrs_v_in_set[0]
                        in_set[u] = False
                        in_set[v] = True
                        conflict_count[neighbors[u]] -= 1
                        conflict_count[neighbors[v]] += 1
                        stuck = 0
                        improved = True
                        continue

            # 3. Try 1-2 Swap (Remove 1, Add 2) -> Size increases by 1
            if not improved and stuck > 50:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    u = in_set_indices[rng.integers(len(in_set_indices))]
                    cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                    if len(cands_v) >= 2:
                        idx = rng.choice(len(cands_v), size=2, replace=False)
                        v1 = cands_v[idx[0]]
                        v2 = cands_v[idx[1]]
                        
                        nbrs_v1 = np.where(conflicts[v1] & in_set)[0]
                        nbrs_v2 = np.where(conflicts[v2] & in_set)[0]
                        
                        if len(nbrs_v1) == 1 and nbrs_v1[0] == u and \
                           len(nbrs_v2) == 1 and nbrs_v2[0] == u and \
                           not conflicts[v1, v2]:
                           
                           in_set[u] = False
                           in_set[v1] = True
                           in_set[v2] = True
                           
                           conflict_count[neighbors[u]] -= 1
                           conflict_count[neighbors[v1]] += 1
                           conflict_count[neighbors[v2]] += 1
                           
                           current_size += 1
                           improved = True
                           stuck = 0
                           if current_size > best_size:
                               best_size = current_size
                               best_in_set = in_set.copy()
                           continue

            if not improved:
                stuck += 1
                if stuck > 100:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    # Run multiple restarts
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Initial Greedy
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Local Search on initial
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.5, 10.0)
        ls_result, ls_size = local_search_2swap(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    # Restarts
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 20:
        n_restarts += 1
        in_set = greedy_init()
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        ls_time = min(remaining_time * 0.4, 5.0)
        ls_result, ls_size = local_search_2swap(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final polish: try to add any remaining vertices with 0 conflicts
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                # Update counts for other not-in-set vertices
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
