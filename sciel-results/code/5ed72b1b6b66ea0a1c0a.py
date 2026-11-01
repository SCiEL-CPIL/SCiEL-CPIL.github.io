"""Kissing number in dimension 11 - Hybrid Local Search and Simulated Annealing."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

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
    
    # Convert to list of lists for faster iteration
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init():
        """Greedy min-degree initialization with randomization."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            # Pick from bottom 20% by degree to balance exploration
            deg_sorted = np.sort(deg[alive_indices])
            threshold = deg_sorted[min(len(alive_indices)-1, int(len(alive_indices)*0.2))]
            cands = alive_indices[deg[alive_indices] <= threshold]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set, max_time=5.0):
        """Local search with 1-add, 1-1 swap, and 1-2 swap."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        no_improve_count = 0
        max_no_improve = 50
        
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
                    no_improve_count = 0
                    continue
            
            # 2. Try 1-1 Swap
            if not improved:
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
                        no_improve_count = 0
                        improved = True
                        continue

            # 3. Try 1-2 Swap (Remove 1, Add 2)
            if not improved:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    # Try multiple random u candidates
                    for _ in range(min(5, len(in_set_indices))):
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
                               no_improve_count = 0
                               if current_size > best_size:
                                   best_size = current_size
                                   best_in_set = in_set.copy()
                               break

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation: remove 2 random vertices to escape local optimum
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 2:
                        idx = rng.choice(len(in_set_indices), size=2, replace=False)
                        for u in in_set_indices[idx]:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                    elif len(in_set_indices) == 1:
                        u = in_set_indices[0]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                    no_improve_count = 0

        return best_in_set, best_size

    def simulated_annealing(initial_solution, time_limit):
        """
        Simulated Annealing with penalty for conflicts.
        Objective: F = Size - P * Conflicts
        This allows temporary violations to escape local minima.
        """
        x = initial_solution.copy()
        
        # Precompute conflict counts for current state
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(x)[0]:
            conflict_count[neighbors[i]] += 1
        
        def get_conflicts(x_state, cc):
            # Total conflicts = 0.5 * sum(cc[i] for i where x[i]=1)
            return np.dot(x_state, cc) / 2.0

        current_conflicts = get_conflicts(x, conflict_count)
        current_size = x.sum()
        
        T = 5.0
        cooling = 0.995
        ls_deadline = time.time() + time_limit
        
        best_x = x.copy()
        best_score = current_size - 10.0 * current_conflicts
        
        while time.time() < ls_deadline:
            i = rng.integers(N)
            
            if x[i]:
                # Try removing i
                # New size: size - 1
                # New conflicts: conflicts decrease by (number of neighbors of i in x) / 2?
                # Actually, if we remove i, the conflicts involving i disappear.
                # The term for i in sum(x*cc) was cc[i].
                # The terms for neighbors j of i were 1 each (since i was in x).
                # So sum(x*cc) decreases by cc[i] + (number of neighbors of i in x).
                # Wait, cc[j] counts neighbors in x. If i is in x, it contributes 1 to cc[j] for all j in neighbors[i].
                # So when we remove i, cc[j] decreases by 1 for all j in neighbors[i].
                # The change in sum(x*cc) is:
                # - x[i]*cc[i] (removed i)
                # - sum_{j in nbrs(i), x[j]=1} x[j] * 1 (j's count decreases by 1)
                # Let k = number of neighbors of i in x.
                # delta_sum = -cc[i] - k
                # delta_conflicts = delta_sum / 2
                nbrs_in_x = np.where(conflicts[i] & x)[0]
                k = len(nbrs_in_x)
                delta_conf = (-conflict_count[i] - k) / 2.0
                new_conf = current_conflicts + delta_conf
                new_size = current_size - 1
                delta_score = (new_size - new_conf) - (current_size - current_conflicts)
                
                if rng.random() < np.exp(delta_score / T):
                    # Accept
                    x[i] = False
                    current_size = new_size
                    current_conflicts = new_conf
                    # Update conflict_count
                    conflict_count[i] = 0 # Not strictly needed if we don't use it for i, but good practice
                    conflict_count[nbrs_in_x] -= 1
                    
            else:
                # Try adding i
                # New size: size + 1
                # New conflicts: increase by (number of neighbors of i in x) / 2
                nbrs_in_x = np.where(conflicts[i] & x)[0]
                k = len(nbrs_in_x)
                delta_conf = k / 2.0
                new_conf = current_conflicts + delta_conf
                new_size = current_size + 1
                delta_score = (new_size - new_conf) - (current_size - current_conflicts)
                
                if rng.random() < np.exp(delta_score / T):
                    # Accept
                    x[i] = True
                    current_size = new_size
                    current_conflicts = new_conf
                    conflict_count[i] = k
                    conflict_count[nbrs_in_x] += 1
            
            T *= cooling
            if T < 0.1:
                T = 0.1
            
            # Track best valid solution or best penalized solution?
            # We want a valid solution at the end.
            # So we track the best solution that has 0 conflicts, or the best penalized one if none found.
            # But SA might end with conflicts. We will clean up later.
            # Let's track the best penalized score to guide the search.
            current_score = current_size - 10.0 * current_conflicts
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                
        # Cleanup: Remove vertices to make it a valid independent set
        # Greedy removal of highest conflict vertices
        x_clean = best_x.copy()
        while True:
            if time.time() > ls_deadline:
                break
            # Find vertex with max conflicts
            # cc[i] for i in x_clean
            active_indices = np.where(x_clean)[0]
            if len(active_indices) == 0:
                break
            cc_active = conflict_count[active_indices]
            # Note: conflict_count might be stale if we didn't update perfectly during SA cleanup, 
            # but we did update it during SA. Let's recalculate for safety if it's fast enough?
            # N is ~1700, so recalc is O(N^2) worst case, but sparse.
            # Let's just use the current conflict_count.
            if np.max(cc_active) <= 0:
                break
            worst_idx = active_indices[np.argmax(cc_active)]
            x_clean[worst_idx] = False
            nbrs = neighbors[worst_idx]
            conflict_count[nbrs] -= 1
            conflict_count[worst_idx] = 0
            
        return x_clean

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
        ls_time = min(remaining_time * 0.4, 10.0)
        ls_result, ls_size = local_search(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    # Restarts with SA
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 50:
        n_restarts += 1
        in_set = greedy_init()
        size = in_set.sum()
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        # Split time between Local Search and SA
        # If greedy start is good, do more LS. If average, try SA to jump.
        if size >= best_size * 0.95:
            ls_time = min(remaining_time * 0.5, 5.0)
            sa_time = min(remaining_time * 0.4, 5.0)
        else:
            ls_time = min(remaining_time * 0.3, 3.0)
            sa_time = min(remaining_time * 0.5, 8.0)
            
        # Phase 1: Local Search
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        current_best_candidate = ls_result
        current_best_size = ls_size
        
        # Phase 2: Simulated Annealing on the LS result
        if time.time() < deadline - 5:
            sa_result = simulated_annealing(current_best_candidate, sa_time)
            sa_size = sa_result.sum()
            if sa_size > current_best_size:
                current_best_candidate = sa_result
                current_best_size = sa_size
                # Quick polish on SA result
                if time.time() < deadline - 3:
                    polish_result, polish_size = local_search(current_best_candidate, max_time=2.0)
                    if polish_size > current_best_size:
                        current_best_candidate = polish_result
                        current_best_size = polish_size

        if current_best_size > best_size:
            best_size = current_best_size
            best_in_set = current_best_candidate.copy()

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
