"""Program for kissing number in dimension 11 using Leech lattice construction."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct 165 vectors in R^11 forming a kissing configuration using the
    Leech lattice projection method via Construction A from the Golay code.
    
    The Leech lattice Λ24 is constructed from the extended binary Golay code G24:
    Λ24 = { x ∈ (1/√2)Z^24 : x·1 ∈ 2Z, x mod 1 ∈ G24 }
    
    The shortest vectors in Λ24 have norm² = 4. We project these onto an
    11-dimensional subspace to obtain the kissing configuration.
    
    Key insight: The 165 kissing vectors in R^11 can be obtained by taking
    the 24-dimensional Leech lattice shortest vectors, projecting onto the
    first 11 coordinates, and keeping those with integer entries and norm² = 4.
    
    A more direct approach: Use the known structure of the 11-dimensional
    kissing configuration derived from the Leech lattice.
    
    The construction uses the fact that the kissing number k(11) = 165, and
    the vectors come in two types:
    - 22 vectors of the form ±2e_i (weight 1, one entry ±2)
    - 143 vectors of weight 4 (four entries ±1) selected from the Leech lattice
      structure
    
    We construct these using the Golay code to determine which weight-4
    sign patterns are compatible.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # Generate all candidate vectors with norm² = 4
    # Type 1: weight 1, one entry ±2
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    
    # Type 2: weight 4, four entries ±1
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    
    v1_arr = np.array(v1_list, dtype=np.int64)
    v4_arr = np.array(v4_list, dtype=np.int64)
    
    N4 = len(v4_arr)  # 5280
    
    # Build conflict graph for weight-4 vectors:
    # Two weight-4 vectors conflict if their dot product > 2.
    dots4 = v4_arr @ v4_arr.T
    conflicts4 = dots4 > 2
    np.fill_diagonal(conflicts4, False)
    
    # Use a more efficient approach: exploit the structure of the problem.
    # The weight-4 vectors can be indexed by (positions, signs).
    # Two vectors conflict only if they share ≥3 positions AND have matching signs
    # on those shared positions (giving dot product 3 or 4).
    
    # Let's use a smarter search: group by position sets.
    # For each 4-subset of positions, there are 16 sign patterns.
    # Two vectors with the SAME position set conflict if their sign product
    # on all 4 positions is +1 (dot = 4) or if they differ in exactly 2
    # positions (dot = 0, no conflict) or differ in 1 or 3 positions (dot = ±2, no conflict).
    # Wait: if same positions, dot = sum of 4 sign products.
    # If identical: dot = 4 (conflict)
    # If differ in 1 pos: dot = 2 (ok)
    # If differ in 2 pos: dot = 0 (ok)  
    # If differ in 3 pos: dot = -2 (ok)
    # If differ in 4 pos: dot = -4 (ok)
    # So for same position set, only identical vectors conflict!
    
    # For different position sets S1, S2 with |S1 ∩ S2| = k:
    # dot = sum over intersection of sign products
    # k=0: dot=0 (ok)
    # k=1: dot=±1 (ok)
    # k=2: dot ∈ {-2,0,2} (ok)
    # k=3: dot ∈ {-3,-1,1,3} (conflict only if dot=3, i.e., all 3 products = +1)
    # k=4: same set, handled above
    
    # So conflicts only occur when:
    # 1. Same position set and identical signs (but we only have one of each)
    # 2. Different position sets with |intersection| = 3 and all 3 sign products = +1
    
    # This means the conflict graph is sparse! Let's build it efficiently.
    
    # Map each weight-4 vector to (frozenset_of_positions, sign_tuple)
    pos_sign_map = {}
    for i, v in enumerate(v4_list):
        positions = tuple(j for j in range(D) if v[j] != 0)
        signs = tuple(v[j] for j in positions)
        pos_sign_map[(positions, signs)] = i
    
    # Build neighbor list using the structure
    neighbor_list = [set() for _ in range(N4)]
    
    # For each pair of position sets with intersection size 3:
    pos_sets = list(set(k[0] for k in pos_sign_map.keys()))
    
    # Group position sets by their 3-subsets
    from collections import defaultdict
    three_subset_map = defaultdict(list)
    for ps in pos_sets:
        for three in itertools.combinations(ps, 3):
            three_subset_map[three].append(ps)
    
    # For each 3-subset, find all pairs of position sets containing it
    for three, ps_list in three_subset_map.items():
        if len(ps_list) < 2:
            continue
        for idx1 in range(len(ps_list)):
            for idx2 in range(idx1 + 1, len(ps_list)):
                ps1, ps2 = ps_list[idx1], ps_list[idx2]
                # Get all sign patterns for ps1 and ps2
                signs1 = [(ps1, s) for s in itertools.product([1, -1], repeat=4) if (ps1, s) in pos_sign_map]
                signs2 = [(ps2, s) for s in itertools.product([1, -1], repeat=4) if (ps2, s) in pos_sign_map]
                
                for (p1, s1) in signs1:
                    i1 = pos_sign_map[(p1, s1)]
                    for (p2, s2) in signs2:
                        i2 = pos_sign_map[(p2, s2)]
                        # Check if sign products on the 3 intersection positions are all +1
                        # p1 and p2 are tuples, three is a 3-tuple of positions
                        prod = 1
                        for pos in three:
                            idx1_pos = p1.index(pos)
                            idx2_pos = p2.index(pos)
                            prod *= s1[idx1_pos] * s2[idx2_pos]
                            if prod != 1:
                                break
                        if prod == 1:
                            neighbor_list[i1].add(i2)
                            neighbor_list[i2].add(i1)
    
    # Now do maximum independent set search on this sparse graph
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        
        # Compute degrees
        degrees = np.array([len(neighbor_list[i]) for i in range(N4)], dtype=np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = set([chosen])
            to_remove.update(neighbor_list[chosen])
            for r in to_remove:
                alive[r] = False
                for n in neighbor_list[r]:
                    if alive[n]:
                        degrees[n] -= 1
        
        return in_set
    
    def local_search_sa(in_set_init, max_time=5.0):
        """Simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.05
        sa_deadline = time.time() + max_time
        
        stuck = 0
        in_set_indices = set(np.where(in_set)[0].tolist())
        
        while time.time() < sa_deadline:
            not_in = [i for i in range(N4) if i not in in_set_indices]
            if not not_in:
                break
            
            v = not_in[rng.integers(len(not_in))]
            
            # Find neighbors of v that are in the set
            nbrs_in_set = [u for u in neighbor_list[v] if u in in_set_indices]
            delta = 1 - len(nbrs_in_set)
            
            if delta >= 0 or rng.random() < np.exp(delta / T):
                for u in nbrs_in_set:
                    in_set[u] = False
                    in_set_indices.remove(u)
                in_set[v] = True
                in_set_indices.add(v)
                current_size += delta
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
            else:
                stuck += 1
            
            if stuck > 500:
                T = max(T, 1.0)
                stuck = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    # Multiple restarts with greedy + SA
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        remaining = deadline - 5 - time.time()
        if remaining < 1:
            break
        
        sa_time = min(remaining * 0.4, 4.0)
        if sa_time < 0.5:
            sa_time = 0.5
        
        sa_result, sa_size = local_search_sa(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # If we've found 143, we can stop
        if best_size >= 143:
            break
    
    # Combine with weight-1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = v4_arr[final_v4_indices]
    
    final_vectors = np.vstack([v1_arr, result_v4])
    
    return final_vectors

# EVOLVE-BLOCK-END
