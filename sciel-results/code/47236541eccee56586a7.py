import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11 such that the minimum 
    pairwise distance is at least the maximum vector norm.
    
    Strategy:
    1. Generate all weight-4 vectors (5280 candidates).
    2. Build conflict graph using triplet matching.
    3. Use min-degree greedy with multiple restarts.
    4. Use aggressive swap-based local search.
    5. Add compatible weight-2 vectors.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    D = 11
    
    # Generate all weight-4 vectors
    vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            vectors.append((support, signs))
            
    n_vecs = len(vectors)
    
    # Build conflict graph using triplet matching
    triplet_map = defaultdict(list)
    
    for idx, (support, signs) in enumerate(vectors):
        for k in range(4):
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            triplet_map[key].append(idx)
    
    neighbors = [set() for _ in range(n_vecs)]
    
    for idx, (support, signs) in enumerate(vectors):
        for k in range(4):
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            for other_idx in triplet_map[key]:
                if other_idx != idx:
                    neighbors[idx].add(other_idx)
    
    degrees = [len(neighbors[i]) for i in range(n_vecs)]
    
    def greedy_min_degree():
        """Greedy that always picks the vertex with minimum degree among remaining."""
        x = [0] * n_vecs
        remaining = set(range(n_vecs))
        
        while remaining and time.time() < deadline - 5.0:
            # Find vertex with minimum degree in remaining subgraph
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg_in_remaining = len(neighbors[v] & remaining)
                if deg_in_remaining < min_deg:
                    min_deg = deg_in_remaining
                    best_v = v
                if min_deg == 0:
                    break
            
            if best_v is None:
                break
            
            x[best_v] = 1
            remaining.discard(best_v)
            for u in neighbors[best_v]:
                remaining.discard(u)
        
        return x
    
    def greedy_sorted(order):
        """Greedy with pre-sorted order"""
        x = [0] * n_vecs
        remaining = set(range(n_vecs))
        
        for v in order:
            if v not in remaining:
                continue
            x[v] = 1
            remaining.discard(v)
            for u in neighbors[v]:
                remaining.discard(u)
        
        return x
    
    def local_search(x, max_time=15.0):
        """Aggressive local search with additions and swaps"""
        x = x[:]
        selected = set(i for i in range(n_vecs) if x[i])
        unselected = set(range(n_vecs)) - selected
        
        start_ls = time.time()
        
        improved = True
        while improved and time.time() < start_ls + max_time and time.time() < deadline - 5.0:
            improved = False
            
            # Try to add any unselected vertex that has no conflicts
            for u in list(unselected):
                if time.time() > deadline - 5.0:
                    break
                if not (neighbors[u] & selected):
                    x[u] = 1
                    selected.add(u)
                    unselected.discard(u)
                    improved = True
                    break
            
            if time.time() > deadline - 5.0:
                break
            
            # Try swaps: remove one selected, add two unselected
            if time.time() < deadline - 10.0 and len(unselected) > 1:
                found_swap = False
                for s in list(selected):
                    if time.time() > deadline - 5.0:
                        break
                    potential_adds = []
                    for u in unselected:
                        if not (neighbors[u] & (selected - {s})):
                            potential_adds.append(u)
                    
                    added = []
                    for u in potential_adds:
                        if time.time() > deadline - 5.0:
                            break
                        if not (neighbors[u] & set(added)):
                            added.append(u)
                        if len(added) == 2:
                            break
                    
                    if len(added) >= 2:
                        x[s] = 0
                        selected.discard(s)
                        unselected.add(s)
                        for u in added:
                            x[u] = 1
                            selected.add(u)
                            unselected.discard(u)
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    continue
            
            # Try 2-for-3 swaps: remove two selected, add three unselected
            if time.time() < deadline - 15.0 and len(unselected) > 2 and len(selected) > 1:
                found_swap2 = False
                s_list = list(selected)
                for i in range(len(s_list)):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i+1, len(s_list)):
                        if time.time() > deadline - 5.0:
                            break
                        s1, s2 = s_list[i], s_list[j]
                        potential_adds = []
                        for u in unselected:
                            if not (neighbors[u] & (selected - {s1, s2})):
                                potential_adds.append(u)
                        
                        added = []
                        for u in potential_adds:
                            if time.time() > deadline - 5.0:
                                break
                            if not (neighbors[u] & set(added)):
                                added.append(u)
                            if len(added) == 3:
                                break
                        
                        if len(added) >= 3:
                            x[s1] = 0
                            x[s2] = 0
                            selected.discard(s1)
                            selected.discard(s2)
                            unselected.add(s1)
                            unselected.add(s2)
                            for u in added:
                                x[u] = 1
                                selected.add(u)
                                unselected.discard(u)
                            improved = True
                            found_swap2 = True
                            break
                    if found_swap2:
                        break
                if found_swap2:
                    continue
        
        return x
    
    # Multiple greedy restarts
    best_x = None
    best_size = 0
    
    rng = random.Random(123)
    
    # First: min-degree greedy
    if time.time() < deadline - 30.0:
        x = greedy_min_degree()
        size = sum(x)
        if size > best_size:
            best_size = size
            best_x = x[:]
    
    # Then: sorted greedy with various orders
    for restart in range(10):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            order = sorted(range(n_vecs), key=lambda i: degrees[i])
        elif restart == 1:
            order = sorted(range(n_vecs), key=lambda i: -degrees[i])
        elif restart == 2:
            order = list(range(n_vecs))
            rng.shuffle(order)
        elif restart == 3:
            order = sorted(range(n_vecs), key=lambda i: (degrees[i], rng.random()))
        elif restart == 4:
            order = sorted(range(n_vecs), key=lambda i: (degrees[i], rng.random()))
        elif restart == 5:
            order = list(range(n_vecs))
            rng.shuffle(order)
        elif restart == 6:
            order = sorted(range(n_vecs), key=lambda i: (degrees[i], rng.random()))
        elif restart == 7:
            order = list(range(n_vecs))
            rng.shuffle(order)
        elif restart == 8:
            order = sorted(range(n_vecs), key=lambda i: (degrees[i], rng.random()))
        else:
            order = list(range(n_vecs))
            rng.shuffle(order)
        
        x = greedy_sorted(order)
        size = sum(x)
        if size > best_size:
            best_size = size
            best_x = x[:]
    
    # Local search on best solution
    if best_x is not None and time.time() < deadline - 20.0:
        remaining_time = min(15.0, deadline - time.time() - 5.0)
        best_x = local_search(best_x, max_time=remaining_time)
    
    # Convert to vector list
    selected_indices = [i for i in range(n_vecs) if best_x[i]] if best_x else []
    
    # Build result vectors from selected weight-4 vectors
    result_vectors = []
    for idx in selected_indices:
        support, signs = vectors[idx]
        v = [0] * D
        for i, s in zip(support, signs):
            v[i] = s
        result_vectors.append(v)
    
    # Try to add weight-2 vectors
    if time.time() < deadline - 5.0:
        w2_vectors = []
        for support in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for i, s in zip(support, signs):
                    v[i] = s
                w2_vectors.append(v)
        
        selected_maps = []
        for idx in selected_indices:
            support, signs = vectors[idx]
            m = {}
            for i, s in zip(support, signs):
                m[i] = s
            selected_maps.append(m)
        
        compatible_w2 = []
        for u in w2_vectors:
            ok = True
            u_supp = [i for i in range(D) if u[i] != 0]
            for m in selected_maps:
                if len(u_supp) == 2:
                    i1, i2 = u_supp
                    if i1 in m and i2 in m:
                        if u[i1] == m[i1] and u[i2] == m[i2]:
                            ok = False
                            break
            if ok:
                compatible_w2.append(u)
        
        result_vectors.extend(compatible_w2)
    
    if not result_vectors:
        for support in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for i, s in zip(support, signs):
                    v[i] = s
                result_vectors.append(v)
                
    result = np.array(result_vectors, dtype=np.int64)
    return result
