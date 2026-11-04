import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Build incompatibility graph as sets for fast intersection
    sup_sets = [set(s) for s in supports]
    incompat = [set() for _ in range(N_sup)]
    
    for i in range(N_sup):
        if time.time() > deadline - 15.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i].add(j)
                incompat[j].add(i)
    
    # Precompute as frozensets for fast intersection
    incompat_frozen = [frozenset(s) for s in incompat]
    
    def greedy_min_degree(available, rng):
        """
        Greedy MIS using minimum degree heuristic.
        """
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 10.0:
                break
            
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(incompat_frozen[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            u = rng.choice(candidates)
            selected.add(u)
            
            to_remove = {u} | (incompat_frozen[u] & avail)
            avail -= to_remove
            
        return selected

    def local_search(selected, rng):
        """
        Try to improve the independent set using 1-2 swaps.
        """
        selected = set(selected)
        improved = True
        stall_count = 0
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            stall_count += 1
            
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for v in sel_list:
                if time.time() > deadline - 10.0:
                    break
                
                sel_minus_v = selected - {v}
                
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (incompat_frozen[x] & sel_minus_v):
                        potential_adds.append(x)
                
                if len(potential_adds) < 2:
                    continue
                
                found = False
                n_pa = len(potential_adds)
                for i in range(n_pa):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pa):
                        u2 = potential_adds[j]
                        if u2 not in incompat_frozen[u1]:
                            selected.discard(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
            
            if not improved:
                if stall_count > 5:
                    break
        
        return selected

    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        avail = set(range(N_sup))
        iset = greedy_min_degree(avail, rng)
        iset = local_search(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
            # Extra local search on improved best
            if time.time() < deadline - 20.0:
                iset = local_search(iset, rng)
                if len(iset) > best_size:
                    best_size = len(iset)
                    best_iset = iset.copy()
        
        # Periodically re-search current best with fresh rng state
        if r % 25 == 24 and best_iset is not None and time.time() < deadline - 15.0:
            improved = local_search(best_iset, rng)
            if len(improved) > best_size:
                best_size = len(improved)
                best_iset = improved
        
    if best_iset is None:
        return np.array([[]], dtype=np.int8)
        
    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
