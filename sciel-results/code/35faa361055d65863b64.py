import numpy as np
import time
from itertools import combinations, product
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330
    
    # Precompute support sets for intersection checks
    sup_sets = [set(s) for s in supports]
    
    # Build incompatibility graph: two supports are incompatible if |intersection| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    
    for i in range(N_sup):
        if time.time() > deadline - 20.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    # Build neighbor lists for faster access
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    
    # Precompute compatibility bitmask: compat[i] is a frozenset of nodes compatible with i
    compat_sets = [set(range(N_sup)) - neighbors[i] - {i} for i in range(N_sup)]
    
    # Precompute: for each node, which nodes are incompatible (for fast blocking check)
    # Use numpy for vectorized degree computation in greedy
    incompat_matrix = incompat  # already computed
    
    def greedy_mis_min_degree(available, rng):
        """Greedy MIS using minimum degree heuristic with numpy acceleration."""
        selected = set()
        avail = set(available)
        avail_arr = np.array(list(avail), dtype=np.int32)
        
        while len(avail_arr) > 0:
            if time.time() > deadline - 10.0:
                break
            
            # Compute degrees within the available set using numpy
            # degree of u = number of neighbors of u that are in avail
            # Use matrix multiplication for speed
            if len(avail_arr) > 100:
                # Fast path: use numpy
                mask = np.zeros(N_sup, dtype=np.bool_)
                mask[avail_arr] = True
                degrees = incompat_matrix[avail_arr] @ mask.astype(np.uint8)
            else:
                # Slow path for small sets
                degrees = np.array([len(neighbors[u] & avail) for u in avail_arr], dtype=np.int32)
            
            min_deg = degrees.min()
            candidates = avail_arr[degrees == min_deg]
            
            # Pick randomly from minimum-degree candidates
            u = int(rng.choice(candidates))
            selected.add(u)
            
            # Remove u and its neighbors from available
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            avail_arr = np.array(list(avail), dtype=np.int32)
            
        return selected
    
    def local_search_1_2_swap(selected, rng):
        """Improve MIS using 1-2 swaps: remove one vertex, add two compatible vertices."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = list(selected)
            
            for v in sel_list:
                if time.time() > deadline - 10.0:
                    break
                
                # Find all nodes compatible with (selected - {v})
                remaining = selected - {v}
                # A node x is a potential add if it's not in remaining, not v, and compatible with all remaining
                potential_adds = []
                for x in compat_sets[v]:  # nodes compatible with v
                    if x in remaining:
                        continue
                    # x must be compatible with all y in remaining
                    if not (neighbors[x] & remaining):
                        potential_adds.append(x)
                
                # Find a pair (u1, u2) in potential_adds that are compatible with each other
                found = False
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if u2 in compat_sets[u1]:
                            selected.discard(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if improved:
                    break  # Restart the outer loop
        
        return selected
    
    def local_search_2_2_swap(selected, rng):
        """Improve MIS using 2-2 swaps: remove two vertices, add two compatible vertices."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = list(selected)
            n_sel = len(sel_list)
            
            for i in range(n_sel):
                if time.time() > deadline - 10.0:
                    break
                v1 = sel_list[i]
                for j in range(i + 1, n_sel):
                    if time.time() > deadline - 10.0:
                        break
                    v2 = sel_list[j]
                    
                    # Find all nodes compatible with (selected - {v1, v2})
                    remaining = selected - {v1, v2}
                    # Potential adds must be compatible with both v1 and v2 (since we're removing them,
                    # they don't need to be compatible with v1/v2, but must be compatible with remaining)
                    potential_adds = []
                    for x in range(N_sup):
                        if x in remaining or x == v1 or x == v2:
                            continue
                        if not (neighbors[x] & remaining):
                            potential_adds.append(x)
                    
                    # Find a pair (u1, u2) in potential_adds that are compatible
                    found = False
                    n_pot = len(potential_adds)
                    for a in range(n_pot):
                        u1 = potential_adds[a]
                        for b in range(a + 1, n_pot):
                            u2 = potential_adds[b]
                            if u2 in compat_sets[u1]:
                                selected.discard(v1)
                                selected.discard(v2)
                                selected.add(u1)
                                selected.add(u2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    
                    if found:
                        break
                if improved:
                    break
        
        return selected
    
    def local_search_add(selected, rng):
        """Try to add vertices that are compatible with all current selected vertices."""
        selected = set(selected)
        
        # Iterate multiple times to allow cascading additions
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            for x in range(N_sup):
                if time.time() > deadline - 10.0:
                    break
                if x in selected:
                    continue
                # Check if x is compatible with all selected
                if not (neighbors[x] & selected):
                    selected.add(x)
                    changed = True
        
        return selected
    
    best_iset = set()
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        if r == 0:
            # First run: pure greedy min-degree
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
        elif r < 15:
            # Greedy + local search
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_2_2_swap(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_add(iset, rng)
        elif r < 30:
            # Perturbed greedy: start with random subset removed
            avail = set(range(N_sup))
            n_remove = rng.integers(10, 50)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_2_2_swap(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_add(iset, rng)
        else:
            # Heavier perturbation + more local search passes
            avail = set(range(N_sup))
            n_remove = rng.integers(20, 80)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_2_2_swap(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_2_2_swap(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_add(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    # Final improvement attempt on best solution
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
        best_iset = local_search_2_2_swap(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
        best_iset = local_search_2_2_swap(best_iset, rng)
        best_iset = local_search_1_2_swap(best_iset, rng)
        best_iset = local_search_add(best_iset, rng)
    
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
        # Fallback: return a small valid set
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
