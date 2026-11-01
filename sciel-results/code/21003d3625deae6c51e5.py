"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
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

    # Group W4 vectors by support
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)
    
    supports = list(support_to_indices.keys())
    n_supports = len(supports)
    
    # Build support-level conflict matrix: conflict if share >= 3 positions
    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True
    
    # Precompute neighbor lists and conflict masks
    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)
    
    # Precompute boolean conflict rows for fast AND operations
    conflict_rows = [support_conflicts[i] for i in range(n_supports)]
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree_supports():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def greedy_max_degree_supports():
        """Greedy independent set with max-degree heuristic."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def random_init_supports(density=0.15):
        """Random initialization with repair."""
        in_set = np.zeros(n_supports, dtype=bool)
        candidates = rng.integers(0, n_supports, size=int(n_supports * density * 3))
        
        for s in candidates:
            if not in_set[s] and not np.any(conflict_rows[s] & in_set):
                in_set[s] = True
        
        return in_set
    
    def fill_independent_set(in_set):
        """Try to add any non-conflicting supports to in_set."""
        polished = True
        while polished and time.time() < deadline - 3:
            polished = False
            not_in = np.where(~in_set)[0]
            for s in not_in:
                if not np.any(conflict_rows[s] & in_set):
                    in_set[s] = True
                    polished = True
        return in_set
    
    def sa_local_search_supports(in_set_init, max_time=10.0):
        """SA with optimized add/remove/swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors_sup[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck_counter = 0
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.65:
                # Add move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                
                # Prefer supports with fewer conflicts
                counts = count_conflicts[not_in_set].astype(float)
                weights = np.exp(-counts / max(T, 0.1))
                wsum = weights.sum()
                if wsum > 0:
                    weights /= wsum
                    s = not_in_set[rng.choice(len(not_in_set), p=weights)]
                else:
                    s = not_in_set[rng.integers(len(not_in_set))]
                
                k = int(count_conflicts[s])
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(conflict_rows[s] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors_sup[u]] -= 1
                    
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors_sup[s]] += 1
                    
            elif move_type < 0.85 and current_size > 0:
                # Remove move
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                    
                s = in_set_indices[rng.integers(len(in_set_indices))]
                accept = rng.random() < np.exp(-1.0 / T)
                
                if accept:
                    in_set[s] = False
                    current_size -= 1
                    count_conflicts[neighbors_sup[s]] -= 1
            else:
                # Swap move: remove one, add one
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                
                s_remove = in_set_indices[rng.integers(len(in_set_indices))]
                s_add = not_in_set[rng.integers(len(not_in_set))]
                
                # Net effect: remove s_remove, add s_add
                # s_add must not conflict with (in_set - {s_remove})
                # conflicts of s_add with in_set minus s_remove
                n_conf = int(np.sum(conflict_rows[s_add] & in_set))
                if s_add != s_remove and conflict_rows[s_add][s_remove]:
                    n_conf -= 1  # s_remove will be gone
                
                if n_conf == 0:
                    # Clean swap
                    in_set[s_remove] = False
                    count_conflicts[neighbors_sup[s_remove]] -= 1
                    in_set[s_add] = True
                    count_conflicts[neighbors_sup[s_add]] += 1
                    # Size unchanged
                else:
                    # Would need to remove n_conf additional nodes
                    # Accept with probability based on net change: -1 (remove) + 1 (add) - n_conf (extra removals)
                    net_delta = 1 - 1 - n_conf
                    if net_delta >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(net_delta / T)
                    
                    if accept:
                        to_remove = np.where(conflict_rows[s_add] & in_set)[0]
                        # Remove s_remove first
                        in_set[s_remove] = False
                        count_conflicts[neighbors_sup[s_remove]] -= 1
                        # Remove conflicting nodes (excluding s_remove which is already gone)
                        to_remove = to_remove[to_remove != s_remove]
                        for u in to_remove:
                            in_set[u] = False
                            count_conflicts[neighbors_sup[u]] -= 1
                        in_set[s_add] = True
                        count_conflicts[neighbors_sup[s_add]] += 1
                        current_size += net_delta
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
            
            if stuck_counter > 3000:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 500
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        strategy = n_restarts % 3
        if strategy == 0:
            in_set = greedy_min_degree_supports()
        elif strategy == 1:
            in_set = greedy_max_degree_supports()
        else:
            in_set = random_init_supports()
            
        # Fill pass
        in_set = fill_independent_set(in_set)
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.4, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
        
        # Fill after SA
        sa_result = fill_independent_set(sa_result)
        sa_size = int(sa_result.sum())
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
    
    # Final fill
    best_support_set = fill_independent_set(best_support_set)
    best_size = int(best_support_set.sum())
    
    # Build final vector list
    selected_support_indices = np.where(best_support_set)[0]
    final_vectors = []
    
    for si in selected_support_indices:
        support = supports[si]
        for idx in support_to_indices[support]:
            final_vectors.append(vectors_v4[idx])
    
    final_vectors.extend(v1_candidates)
    
    result = np.array(final_vectors, dtype=np.int64)
    
    if len(result) == 0:
        all_supports = list(itertools.combinations(range(D), 4))
        selected_supports = []
        for s in all_supports:
            valid = True
            for t in selected_supports:
                if len(set(s).intersection(t)) > 2:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
                if len(selected_supports) >= 32:
                    break
        
        fallback_vectors = []
        for s in selected_supports:
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                fallback_vectors.append(v)
        
        for pos in range(D):
            for sign in [1, -1]:
                v = [0] * D
                v[pos] = 2 * sign
                fallback_vectors.append(v)
        
        result = np.array(fallback_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
