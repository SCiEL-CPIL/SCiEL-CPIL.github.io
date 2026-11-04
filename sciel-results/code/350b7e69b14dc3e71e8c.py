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
    
    def greedy_mis_min_degree(available, rng):
        """Greedy MIS using minimum degree heuristic."""
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 10.0:
                break
            
            # Find node with minimum degree in induced subgraph
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(neighbors[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            # Pick randomly from minimum-degree candidates
            u = rng.choice(candidates)
            selected.add(u)
            
            # Remove u and its neighbors
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected
    
    def local_search_1_2_swap(selected, rng):
        """Improve MIS using 1-2 swaps: remove one vertex, add two compatible vertices."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            for v in list(selected):
                if time.time() > deadline - 10.0:
                    break
                
                # Find all nodes compatible with (selected - {v})
                remaining = selected - {v}
                potential_adds = []
                for x in range(N_sup):
                    if x in remaining or x == v:
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
                            if incompat[u1, u2] == 0:
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
        
        for x in range(N_sup):
            if time.time() > deadline - 10.0:
                break
            if x in selected:
                continue
            # Check if x is compatible with all selected
            if not (neighbors[x] & selected):
                selected.add(x)
        
        return selected
    
    best_iset = set()
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        if r == 0:
            # First run: pure greedy min-degree
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
        elif r < 10:
            # Greedy + local search
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_2_2_swap(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        else:
            # Perturbed greedy: start with random subset removed
            avail = set(range(N_sup))
            # Remove a random fraction of nodes to break symmetry
            if r > 10:
                n_remove = rng.integers(10, 50)
                to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
                avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_2_2_swap(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
        
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
