import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Constructive approach: D_11^+ lattice shortest vectors
    # D_n = {x in Z^n : sum(x_i) even}
    # D_n^+ = D_n union {(x_i + 1/2) : x_i in Z^n}
    # Shortest vectors in D_11^+ have norm^2 = 4:
    #   Type A: exactly 4 nonzero coords in {+1,-1}, rest 0, with even sum
    #   Type B: all 11 coords are +/-1/2 (these are in D_11^+ \ D_11)
    #           But we need integer vectors, so Type B are (1/2, ..., 1/2) * 2 = (1,...,1)
    #           Wait - we need integer vectors. Let me reconsider.
    #
    # Actually for integer vectors with norm^2 = 4:
    #   - Weight 4: four +/-1 entries, rest 0. Sum can be -4,-2,0,2,4 (all even).
    #     All are in D_11. Count: C(11,4)*2^4 = 330*16 = 5280
    #   - Weight 1: one +/-2 entry, rest 0. Sum is +/-2 (even). In D_11.
    #     Count: 11*2 = 22
    #   - Weight 2: two entries, values from {+1,-1,+2,-2} with sum of squares = 4
    #     Only possibility: (+1,+1) or (+1,-1) etc. -> 1+1=2 != 4. Nope.
    #     Actually: a^2 + b^2 = 4 with a,b in integers: (0,2),(2,0),(0,-2),(-2,0),(1,? ) 
    #     1+3 no, 4+0 yes. So only the weight-1 case with value 2.
    #
    # So the full set of integer vectors with norm^2 = 4 is:
    #   Type A: weight 4, entries in {-1,0,+1}: C(11,4) * 2^4 = 5280
    #   Type B: weight 1, entry in {-2,+2}: 22
    #   Total: 5302
    #
    # For the kissing configuration, we need min pairwise distance >= max norm = 2.
    # Equivalently: for all pairs u,v, ||u-v||^2 >= 4, i.e., u.v <= (||u||^2 + ||v||^2 - 4)/2 = (4+4-4)/2 = 2.
    # So we need an independent set in the graph where edge exists if u.v > 2.
    #
    # The D_11^+ lattice kissing number is known to be 2304.
    # The 2304 shortest vectors of D_11^+ (norm^2=4) form the kissing configuration.
    # These are a subset of our 5302 vectors.
    #
    # Let's generate all 5302 and find the maximum independent set where u.v <= 2 for all pairs.
    # But 5302 vertices is too large for brute force. Let's use the structure.
    #
    # Actually, let me reconsider. The task says "minimum pairwise distance is at least the maximum vector norm."
    # If all vectors have norm^2 = 4, then max norm = 2.
    # We need min pairwise distance >= 2, i.e., min ||u-v||^2 >= 4.
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 8 - 2*u.v >= 4
    # => u.v <= 2
    #
    # For Type A vectors (weight 4, entries +/-1): u.v is the number of matching signs minus mismatching signs
    #   on overlapping positions. Range: -4 to 4.
    # For Type A vs Type B: u.v = 2 * (sign of u at pos) * (sign of b) if they share a position, else 0.
    #   So u.v is in {-2, 0, 2}.
    # For Type B vs Type B: u.v = 4 if same, 0 if different positions, -4 if opposite.
    #
    # The kissing number of D_11^+ is 2304. Let me construct it properly.
    #
    # The 2304 vectors are:
    #   - 11 * 2 = 22 vectors of type (±2, 0, ..., 0)
    #   - The remaining 2282 are weight-4 vectors with specific sign patterns.
    #
    # Actually, let me just generate all candidates and use a fast greedy + local search.
    # With 5302 vectors, the conflict matrix is 5302x5302 ~ 28M entries, manageable.

    # Generate all vectors with norm^2 = 4
    all_vectors = []
    
    # Type A: weight 4, entries in {-1, 0, 1}
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            all_vectors.append(v)
    
    # Type B: weight 1, entry in {-2, 2}
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            all_vectors.append(v)
    
    vectors = np.array(all_vectors, dtype=np.int64)
    N = len(vectors)
    
    # Verify all have norm^2 = 4
    norms_sq = np.sum(vectors * vectors, axis=1)
    assert np.all(norms_sq == 4), f"Norm check failed: {set(norms_sq.tolist())}"
    
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors @ vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        return in_set
    
    def local_search_fast(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        not_in_set = list(np.where(~in_set)[0])
        pos_in_not_in_set = {}
        for idx, v in enumerate(not_in_set):
            pos_in_not_in_set[v] = idx
        
        def remove_from_not_in_set(v):
            if v in pos_in_not_in_set:
                idx = pos_in_not_in_set[v]
                last = not_in_set[-1]
                not_in_set[idx] = last
                pos_in_not_in_set[last] = idx
                not_in_set.pop()
                del pos_in_not_in_set[v]
        
        def add_to_not_in_set(v):
            if v not in pos_in_not_in_set:
                pos_in_not_in_set[v] = len(not_in_set)
                not_in_set.append(v)
        
        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break
            
            if len(not_in_set) > 500:
                sample_size = 500
                sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                sample = np.array([not_in_set[i] for i in sample_indices])
            else:
                sample = np.array(not_in_set)
            
            counts = count_conflicts[sample]
            min_count = int(np.min(counts))
            
            if min_count == 0:
                zero_mask = counts == 0
                if np.any(zero_mask):
                    zero_cands = sample[zero_mask]
                    v = int(zero_cands[rng.integers(len(zero_cands))])
                else:
                    v = int(not_in_set[rng.integers(len(not_in_set))])
                
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue
            
            one_mask = counts == 1
            if np.any(one_mask) and rng.random() < 0.7:
                one_cands = sample[one_mask]
                v = int(one_cands[rng.integers(len(one_cands))])
                k = 1
                delta = 1 - k
                accept = rng.random() < min(1.0, 0.3 + T * 0.5)
            else:
                min_mask = counts == min_count
                min_cands = sample[min_mask]
                v = int(min_cands[rng.integers(len(min_cands))])
                k = int(count_conflicts[v])
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    if delta < -20 * T:
                        accept = False
                    else:
                        accept = rng.random() < np.exp(delta / T)
            
            k = int(count_conflicts[v])
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)
            
            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                        add_to_not_in_set(int(u))
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 800:
                T = max(T, 2.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    def polish_pass(in_set_init, max_time=10.0):
        """Pure-add polish: only add vertices with zero conflicts to the set.
        This monotonically increases size and never degrades."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        polish_deadline = time.time() + max_time
        
        # Build conflict count for each vertex against the current set
        count_conflicts = np.zeros(N, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        # Repeatedly find and add all zero-conflict vertices
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            zero_cands = np.where(count_conflicts == 0)[0]
            zero_cands = zero_cands[~in_set[zero_cands]]
            if len(zero_cands) == 0:
                break
            # Add all zero-conflict vertices at once
            for v in zero_cands:
                in_set[v] = True
                count_conflicts[neighbors[v]] += 1
            current_size = int(in_set.sum())
            improved = True
        
        return in_set, current_size

    # Main Loop
    n_restarts = 0
    max_restarts = 80

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 2:
            break
        
        sa_time = min(remaining_time * 0.50, 16.0)
        if sa_time < 2:
            sa_time = 2.0
        
        sa_result, sa_size = local_search_fast(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polish_time = min(remaining_time * 0.40, 10.0)
            if polish_time > 1:
                polish_result, polish_size = polish_pass(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
                
                # Second SA pass on the polished result for further improvement
                remaining_time2 = deadline - 5 - time.time()
                if remaining_time2 > 3:
                    sa2_time = min(remaining_time2 * 0.50, 10.0)
                    if sa2_time > 1:
                        sa2_result, sa2_size = local_search_fast(
                            polish_result, max_time=sa2_time, min_T=0.03, cooling=0.9997
                        )
                        if sa2_size > best_size:
                            best_size = sa2_size
                            best_in_set = sa2_result.copy()
                        
                        # Final polish
                        remaining_time3 = deadline - 4 - time.time()
                        if remaining_time3 > 1:
                            final_polish, final_size = polish_pass(sa2_result, max_time=min(remaining_time3 * 0.5, 5.0))
                            if final_size > best_size:
                                best_size = final_size
                                best_in_set = final_polish.copy()
    
    # Final safety polish on best result
    remaining_final = deadline - 2 - time.time()
    if remaining_final > 0.5:
        final_polish, final_size = polish_pass(best_in_set, max_time=min(remaining_final * 0.8, 5.0))
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_polish.copy()
    
    final_indices = np.where(best_in_set)[0]
    result = vectors[final_indices]
    
    return result

# EVOLVE-BLOCK-END
