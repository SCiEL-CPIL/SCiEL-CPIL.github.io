import numpy as np
import itertools
import time

def construct():
    """
    Returns a list of integer vectors in R^11 forming a kissing configuration.
    Uses the D11 lattice shortest vectors (squared norm 4, even coordinate sum).
    These form a valid kissing configuration with minimum pairwise distance squared = 4,
    which equals the maximum squared norm (4), satisfying the constraint.
    
    The D11 lattice consists of integer vectors with even coordinate sum.
    The shortest vectors have squared norm 4 and come in two types:
    1. Weight 1: one coordinate is ±2, rest 0 (11 * 2 = 22 vectors)
    2. Weight 2: two coordinates are ±1, rest 0 (C(11,2) * 4 = 220 vectors)
    
    Total: 242 vectors. But wait, let me verify the minimum distance.
    
    Actually, for the D_n lattice, the shortest vectors are:
    - ±e_i ± e_j for i≠j (weight 2, norm^2=2) -- but these have odd coordinate sum?
      No: e_i + e_j has sum 2 (even), e_i - e_j has sum 0 (even). So these are in D_n.
    - ±2e_i (weight 1, norm^2=4) -- sum is ±2 (even). These are in D_n.
    
    Wait, I need to be more careful. The D_n lattice is {x in Z^n : sum(x_i) is even}.
    Shortest vectors in D_n:
    - For n >= 3: the minimal norm is 2 (squared norm 2), from vectors like e_i + e_j or e_i - e_j.
    - But e_i + e_j has coordinate sum 2 (even), so it's in D_n.
    - e_i - e_j has coordinate sum 0 (even), so it's in D_n.
    - 2e_i has coordinate sum 2 (even), so it's in D_n, but norm^2 = 4.
    
    So the shortest vectors in D_11 have norm^2 = 2, not 4.
    
    Let me reconsider. The task says "minimum pairwise distance is at least the maximum vector norm."
    If all vectors have norm^2 = 2, then max norm = sqrt(2).
    We need min pairwise distance >= sqrt(2), i.e., min distance^2 >= 2.
    
    For weight-2 vectors (norm^2=2):
    - Two vectors sharing 1 index: dot product is ±1. Distance^2 = 2+2-2*dot = 2∓2.
      If dot=1: dist^2=0 (same vector, excluded). If dot=-1: dist^2=4.
      Wait, let me recompute. If v and w share index i:
      v_i = a, w_i = b where a,b in {±1}.
      They share exactly one index i, and differ on another.
      v = (a, 0, ..., 0, c, 0, ...) with a at pos i, c at pos j.
      w = (b, 0, ..., 0, d, 0, ...) with b at pos i, d at pos k (k≠j).
      dot = a*b.
      dist^2 = ||v||^2 + ||w||^2 - 2*dot = 2 + 2 - 2ab = 4 - 2ab.
      If ab=1: dist^2=2. If ab=-1: dist^2=6.
      So min dist^2 = 2, which equals max norm^2 = 2. Valid!
    
    So the 220 weight-2 vectors form a valid kissing configuration with k=220.
    
    Can we do better? The inspirations suggest using independent set search on weight-4 vectors.
    Let's try combining: use the D11 lattice structure but search for larger configurations.
    
    Actually, the key insight from inspirations is to use simulated annealing / greedy independent set
    on a larger candidate pool to find more vectors that satisfy the distance constraint.
    
    Let me use the approach from inspiration 1: generate weight-4 vectors and find a large
    independent set where no two have dot product > 2 (since norm^2=4, we need dist^2 >= 4,
    i.e., ||v-w||^2 = 4+4-2*dot >= 4, so dot <= 2).
    
    Additionally, we can include weight-1 vectors (norm^2=4) and check compatibility.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
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
        
    # Compute conflict matrix: conflict if dot > 2
    # For two weight-4 vectors, dot product ranges from -4 to 4.
    # We need dist^2 >= 4, so 4+4-2*dot >= 4, i.e., dot <= 2.
    # Conflict if dot > 2, i.e., dot in {3, 4}.
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
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
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final result: include all W1 vectors (they are compatible with each other and with W4)
    # Check: W1 vector has one entry ±2. W4 vector has four entries ±1.
    # dot product: if W1's nonzero position matches one of W4's nonzero positions,
    # dot = ±2*±1 = ±2. dist^2 = 4+4-2*(±2) = 8∓4 = 4 or 8. Both >= 4. OK.
    # If no match, dot = 0, dist^2 = 8. OK.
    # W1-W1: two W1 vectors. If same position, dot = ±4, dist^2 = 4+4-2*(±4) = 0 or 16.
    #   Same sign: dist^2=0 (same vector, excluded). Opposite sign: dist^2=16. OK.
    # If different positions, dot=0, dist^2=8. OK.
    
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
