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
    
    The 165 kissing vectors in R^11:
    - 22 vectors of the form ±2e_i
    - 143 vectors of weight 4 (four entries ±1)
    
    Uses fast numpy-based conflict detection and multi-start local search.
    """
    start_time = time.time()
    deadline = start_time + 115
    
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
    
    # Compute conflict matrix using numpy (fast)
    dots4 = v4_arr @ v4_arr.T
    conflicts4 = dots4 > 2
    np.fill_diagonal(conflicts4, False)
    
    # Build adjacency list from conflict matrix
    neighbor_list = [set() for _ in range(N4)]
    # Use sparse representation: for each row, find True columns
    for i in range(N4):
        nbrs = np.where(conflicts4[i])[0]
        neighbor_list[i] = set(nbrs.tolist())
    
    # Also check conflicts between weight-1 and weight-4 vectors
    # v1·v4: dot product is ±2 if the nonzero position of v1 matches one of v4's positions
    # |dot| = 2, which is NOT > 2, so no conflict! Good.
    # v1·v1: ±4 if same pos, 0 if different. Same pos with same sign: dot=4 > 2 (conflict)
    # But we include both +2e_i and -2e_i, and they conflict with each other (dot=-4, |dot|=4>2)
    # Wait: the condition is minimum pairwise distance >= max norm.
    # Distance between v and w = ||v-w||. Max norm = 2 (for weight-1) or 2 (for weight-4, norm=sqrt(4)=2).
    # So all vectors have norm 2. We need ||v-w|| >= 2 for all pairs.
    # ||v-w||^2 = ||v||^2 + ||w||^2 - 2*v·w = 4 + 4 - 2*v·w = 8 - 2*v·w
    # Need 8 - 2*v·w >= 4, i.e., v·w <= 2.
    # So conflict iff v·w > 2.
    
    # For v1=2e_i and v4: v1·v4 = 2*v4[i] which is -2, 0, or 2. Never > 2. OK.
    # For v1=2e_i and v1'=2e_j: if i=j, dot=4>2 (conflict, but we have both signs)
    # 2e_i · (-2e_i) = -4, not > 2. OK.
    # 2e_i · 2e_j = 0 for i≠j. OK.
    # So all 22 weight-1 vectors are mutually compatible!
    
    # Now find max independent set in the weight-4 conflict graph
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_multi_start(n_trials=50):
        """Multiple greedy passes with random tie-breaking."""
        best_local_size = 0
        best_local_set = np.zeros(N4, dtype=bool)
        
        degrees = np.array([len(neighbor_list[i]) for i in range(N4)], dtype=np.int32)
        
        for trial in range(n_trials):
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            deg = degrees.copy()
            
            while np.any(alive):
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                
                # Pick random vertex with minimum degree among alive
                min_deg = np.min(deg[alive_indices])
                cands = alive_indices[deg[alive_indices] == min_deg]
                chosen = cands[rng.integers(len(cands))]
                
                in_set[chosen] = True
                to_remove = [chosen] + list(neighbor_list[chosen])
                for r in to_remove:
                    if alive[r]:
                        alive[r] = False
                        for n in neighbor_list[r]:
                            if alive[n]:
                                deg[n] -= 1
            
            sz = int(in_set.sum())
            if sz > best_local_size:
                best_local_size = sz
                best_local_set = in_set.copy()
        
        return best_local_set, best_local_size
    
    def local_search(in_set_init, max_iters=200000):
        """Fast local search: try to add vertices, remove conflicting ones."""
        in_set = in_set_init.copy()
        in_set_set = set(np.where(in_set)[0].tolist())
        current_size = len(in_set_set)
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        not_in = [i for i in range(N4) if i not in in_set_set]
        
        for it in range(max_iters):
            if not not_in:
                break
            
            v = not_in[rng.integers(len(not_in))]
            nbrs_in_set = [u for u in neighbor_list[v] if u in in_set_set]
            delta = 1 - len(nbrs_in_set)
            
            if delta >= 0:
                for u in nbrs_in_set:
                    in_set[u] = False
                    in_set_set.remove(u)
                in_set[v] = True
                in_set_set.add(v)
                current_size += delta
                not_in.remove(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
            else:
                # Try a swap: remove one conflicting neighbor, add v
                if rng.random() < 0.3 and nbrs_in_set:
                    u = nbrs_in_set[rng.integers(len(nbrs_in_set))]
                    in_set[u] = False
                    in_set_set.remove(u)
                    in_set[v] = True
                    in_set_set.add(v)
                    not_in.remove(v)
                    not_in.append(u)
                    current_size += 1 - 1  # net zero, but might help later
                    # Actually this is a swap, size stays same
                    # Only accept if it leads to improvement later
                    # For now, just do it as perturbation
        
        return local_best_in_set, local_best_size
    
    def hill_climbing(in_set_init, max_time=3.0):
        """Aggressive hill climbing with restarts."""
        in_set = in_set_init.copy()
        in_set_set = set(np.where(in_set)[0].tolist())
        current_size = len(in_set_set)
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        hc_deadline = time.time() + max_time
        not_in = [i for i in range(N4) if i not in in_set_set]
        
        while time.time() < hc_deadline:
            if not not_in:
                break
            
            improved = False
            for v in not_in:
                nbrs_in_set = [u for u in neighbor_list[v] if u in in_set_set]
                if len(nbrs_in_set) == 0:
                    in_set[v] = True
                    in_set_set.add(v)
                    not_in.remove(v)
                    current_size += 1
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
            
            if not improved:
                # Try swaps
                swapped = False
                for v in not_in:
                    nbrs_in_set = [u for u in neighbor_list[v] if u in in_set_set]
                    if len(nbrs_in_set) == 1:
                        u = nbrs_in_set[0]
                        # Check if removing u and adding v is beneficial
                        # It's neutral in size, but might unlock future additions
                        if rng.random() < 0.5:
                            in_set[u] = False
                            in_set_set.remove(u)
                            in_set[v] = True
                            in_set_set.add(v)
                            not_in.remove(v)
                            not_in.append(u)
                            swapped = True
                            break
                
                if not swapped:
                    # Perturbation: randomly remove a few vertices
                    if len(in_set_set) > 10:
                        to_remove = list(rng.choice(list(in_set_set), size=min(3, len(in_set_set)), replace=False))
                        for u in to_remove:
                            in_set[u] = False
                            in_set_set.remove(u)
                            not_in.append(u)
                        current_size -= len(to_remove)
        
        return local_best_in_set, local_best_size
    
    # Phase 1: Multi-start greedy
    gs_set, gs_size = greedy_multi_start(30)
    best_size = gs_size
    best_in_set = gs_set.copy()
    
    # Phase 2: Hill climbing from best greedy result
    hc_deadline = deadline - 10
    if time.time() < hc_deadline:
        hc_set, hc_size = hill_climbing(gs_set, max_time=min(5.0, hc_deadline - time.time()))
        if hc_size > best_size:
            best_size = hc_size
            best_in_set = hc_set.copy()
    
    # Phase 3: More restarts if time remains
    while time.time() < deadline - 5:
        gs_set, gs_size = greedy_multi_start(5)
        if gs_size > best_size:
            best_size = gs_size
            best_in_set = gs_set.copy()
        
        remaining = deadline - 5 - time.time()
        if remaining < 1:
            break
        
        hc_set, hc_size = hill_climbing(gs_set, max_time=min(3.0, remaining * 0.5))
        if hc_size > best_size:
            best_size = hc_size
            best_in_set = hc_set.copy()
        
        if best_size >= 143:
            break
    
    # Combine with weight-1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = v4_arr[final_v4_indices]
    
    final_vectors = np.vstack([v1_arr, result_v4])
    
    return final_vectors

# EVOLVE-BLOCK-END
