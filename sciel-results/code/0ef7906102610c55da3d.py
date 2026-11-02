import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 112

    # Generate all weight-4 vectors in D_11 (even sum of coordinates)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            s = sum(signs)
            if s % 2 == 0:  # even sum => in D_11
                v = [0] * D
                for pos, sg in zip(positions, signs):
                    v[pos] = sg
                v4_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Also add weight-2 vectors (±2, 0,...,0) which have norm²=4
    v2_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v2_candidates.append(v)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    N2 = len(vectors_v2)
    
    # Combine all candidates
    all_vectors = np.vstack([vectors_v4, vectors_v2])
    N = len(all_vectors)
    
    # Build conflict graph: conflict if dot > 2 (for norm²=4 vectors)
    dots = all_vectors @ all_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as list of arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Greedy min-degree independent set with random tie-breaking
    rng = np.random.default_rng(42)
    in_set = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    degs = degrees.copy()
    
    while np.any(alive) and time.time() < deadline - 15:
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break
        d = degs[alive_idx]
        min_d = np.min(d)
        c = alive_idx[d == min_d]
        v = c[rng.integers(len(c))]
        in_set[v] = True
        nbrs = neighbors[v]
        to_rem = np.unique(np.concatenate(([v], nbrs)))
        alive[to_rem] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degs[rem] -= conflicts[rem][:, to_rem].sum(axis=1)
    
    current_size = int(in_set.sum())
    
    # SA local search with efficient conflict counting
    count_conflicts = np.zeros(N, dtype=np.int32)
    in_set_indices = np.where(in_set)[0]
    for i in in_set_indices:
        count_conflicts[neighbors[i]] += 1
    
    local_best_size = current_size
    local_best_in_set = in_set.copy()
    T = 1.0
    cooling = 0.9997
    min_T = 0.001
    stuck = 0
    
    # Precompute which vertices are not in set for faster sampling
    sa_deadline = deadline - 3
    while time.time() < sa_deadline:
        not_in_set = np.where(~in_set)[0]
        if len(not_in_set) == 0:
            break
        
        if rng.random() < 0.75:
            # Try to add a vertex with fewest conflicts
            counts = count_conflicts[not_in_set]
            min_c = np.min(counts)
            cands = not_in_set[counts == min_c]
            v = cands[rng.integers(len(cands))]
            k = int(min_c)
            delta = 1 - k
            if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                to_remove = np.where(conflicts[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
        else:
            # Try to remove a random vertex from the set
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) == 0:
                break
            v = in_set_indices[rng.integers(len(in_set_indices))]
            delta = -1
            if T > min_T and rng.random() < np.exp(delta / T):
                in_set[v] = False
                current_size += delta
                count_conflicts[neighbors[v]] -= 1
        
        if current_size > local_best_size:
            local_best_size = current_size
            local_best_in_set = in_set.copy()
            stuck = 0
        else:
            stuck += 1
        if stuck > 3000:
            T = max(T, 1.0)
            stuck = 0
        T = max(min_T, T * cooling)
    
    # Final polish: try to add any vertex with zero conflicts
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(N):
            if not in_set[i] and count_conflicts[i] == 0:
                in_set[i] = True
                current_size += 1
                count_conflicts[neighbors[i]] += 1
                changed = True
                local_best_size = max(local_best_size, current_size)
    
    if local_best_size > current_size:
        current_size = local_best_size
        in_set = local_best_in_set.copy()
    
    final_vectors = all_vectors[in_set]
    return final_vectors
