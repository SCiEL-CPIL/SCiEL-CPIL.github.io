import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # E8 x D3 lattice construction with mixed vectors
    # E8 part: 240 vectors of norm 4 (weight 4, ±1 entries)
    # D3 part: 12 vectors of norm 2 (weight 2, ±1 entries in last 3 dims)
    # Mixed: vectors with norm sqrt(8) that have entries in both E8 and D3 parts

    # First, generate the standard E8 x D3 lattice vectors
    # E8: all weight-4 vectors with ±1 entries
    e8_vectors = []
    for support in combinations(range(8), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            e8_vectors.append(v)
    
    # D3: all weight-2 vectors in dimensions 8,9,10
    d3_vectors = []
    for support in combinations(range(8, 11), 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            d3_vectors.append(v)
    
    # Mixed vectors: weight-2 vectors with one entry in dims 0-7 and one in dims 8-10
    mixed_vectors = []
    for i in range(8):
        for j in range(8, 11):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = si
                    v[j] = sj
                    mixed_vectors.append(v)
    
    # Total: 240 + 12 + 96 = 348 vectors
    all_vectors = e8_vectors + d3_vectors + mixed_vectors
    
    # Verify the constraint
    if len(all_vectors) > 0:
        arr = np.array(all_vectors, dtype=np.int32)
        max_norm_sq = np.max(np.sum(arr**2, axis=1))
        # Compute all pairwise distances squared
        min_dist_sq = float('inf')
        for i in range(len(arr)):
            if time.time() > deadline - 5.0:
                break
            diffs = arr[i+1:] - arr[i]
            dists_sq = np.sum(diffs**2, axis=1)
            if len(dists_sq) > 0:
                min_dist_sq = min(min_dist_sq, np.min(dists_sq))
        
        if min_dist_sq >= max_norm_sq:
            return np.array(all_vectors, dtype=np.int8)
    
    # Fallback: use MIS-based approach if verification fails
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330
    
    # Precompute support sets for intersection checks
    sup_sets = [set(s) for s in supports]
    
    # Build incompatibility graph: two supports are incompatible if |intersection| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    # Build neighbor lists for faster access
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    incompat_bool = incompat.astype(bool)
    
    def greedy_mis_min_degree(available, rng):
        """Greedy MIS using minimum degree heuristic."""
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            if len(avail) > 200:
                avail_arr = np.array(list(avail), dtype=np.int32)
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = deg_arr.min()
                min_mask = deg_arr == min_deg
                candidates = avail_arr[min_mask]
                u = int(rng.choice(candidates))
            else:
                min_deg = float('inf')
                candidates = []
                for u in avail:
                    d = len(neighbors[u] & avail)
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                u = rng.choice(candidates)
            
            selected.add(u)
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
        
        return selected
    
    def local_search_add(selected, rng):
        """Try to add vertices that are compatible with all current selected vertices."""
        selected = set(selected)
        
        if not selected:
            return selected
        
        sel_arr = np.array(list(selected), dtype=np.int32)
        compatible_mask = ~incompat_bool[:, sel_arr].any(axis=1)
        compatible_mask[sel_arr] = False
        
        addable = np.where(compatible_mask)[0]
        for x in addable:
            if time.time() > deadline - 10.0:
                break
            selected.add(int(x))
        
        return selected
    
    def local_search_1_2_swap(selected, rng):
        """Improve MIS using 1-2 swaps."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            for v in list(selected):
                if time.time() > deadline - 10.0:
                    break
                
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)
                
                found = False
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if incompat[u1, u2] == 0:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if improved:
                    break
        
        return selected
    
    best_iset = set()
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    n_restarts = 120
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if r == 0:
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
        elif r < 40:
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        elif r < 80:
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 30)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        else:
            avail = set(range(N_sup))
            n_remove = rng.integers(20, 60)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    # Final improvement attempts on best solution
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    if time.time() < deadline - 5.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    # Generate vectors: for each support in the MIS, include all 2^4 = 16 sign patterns
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    if not vectors:
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
