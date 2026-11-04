import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    supports = list(combinations(range(D), 4))
    N = len(supports)
    sup_sets = [set(s) for s in supports]
    
    # Bitset representation
    # compat_mask[i] has bit j set if i and j are compatible (intersection < 3)
    compat_mask = [0] * N
    for i in range(N):
        mask = 0
        for j in range(N):
            if i == j:
                continue
            if len(sup_sets[i] & sup_sets[j]) < 3:
                mask |= (1 << j)
        compat_mask[i] = mask
        
    def greedy_mis(rng):
        avail_mask = (1 << N) - 1
        selected = []
        while avail_mask:
            # Find vertex in avail with min degree in avail
            min_deg = N
            best_v = -1
            # Iterate bits in avail_mask
            # To speed up, we can just pick a random one if N is small, 
            # or iterate. N=330, iteration is fine.
            candidates = []
            for v in range(N):
                if (avail_mask >> v) & 1:
                    # Count neighbors in avail
                    # neighbors = compat_mask[v] & avail_mask
                    # deg = bit_count(neighbors)
                    # Wait, compat_mask[v] contains compatible nodes.
                    # Degree in conflict graph = N - 1 - bit_count(compat_mask[v])
                    # But we need degree in the INDUCED subgraph of avail.
                    # Conflict neighbors in avail = avail & ~compat_mask[v] & ~(1<<v)
                    # deg = bit_count(avail & ~compat_mask[v])
                    deg = (avail & ~compat_mask[v]).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
            if best_v == -1: break
            selected.append(best_v)
            # Remove best_v and its conflict neighbors from avail
            # Conflict neighbors = ~compat_mask[best_v]
            # Keep nodes compatible with best_v
            avail &= compat_mask[best_v]
            avail &= ~(1 << best_v)
        return set(selected)

    def local_search(iset, rng):
        iset = set(iset)
        iset_mask = 0
        for v in iset:
            iset_mask |= (1 << v)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try 1-swap: Remove u, add MIS of candidates
            for u in list(iset):
                # Remove u
                iset.remove(u)
                iset_mask &= ~(1 << u)
                
                # Candidates compatible with iset
                # compat_with_iset = ALL(compat_mask[v] for v in iset)
                # If iset is empty, all are candidates
                if iset:
                    # Start with all ones
                    c_mask = (1 << N) - 1
                    for v in iset:
                        c_mask &= compat_mask[v]
                else:
                    c_mask = (1 << N) - 1
                
                # Exclude iset itself
                c_mask &= ~iset_mask
                
                # Find MIS of c_mask
                # Greedy on c_mask
                c_avail = c_mask
                c_selected = []
                while c_avail:
                    min_deg = N
                    best_v = -1
                    for v in range(N):
                        if (c_avail >> v) & 1:
                            deg = (c_avail & ~compat_mask[v]).bit_count()
                            if deg < min_deg:
                                min_deg = deg
                                best_v = v
                    if best_v == -1: break
                    c_selected.append(best_v)
                    c_avail &= compat_mask[best_v]
                    c_avail &= ~(1 << best_v)
                    
                if len(c_selected) > 1:
                    for v in c_selected:
                        iset.add(v)
                        iset_mask |= (1 << v)
                    improved = True
                    break
                else:
                    # No improvement, restore u
                    iset.add(u)
                    iset_mask |= (1 << u)
                    
        return iset

    best_iset = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 10.0:
        iset = greedy_mis(rng)
        iset = local_search(iset, rng)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset = iset.copy()
            
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    for i in range(D):
        for s in [2, -2]:
            v = np.zeros(D, dtype=np.int8)
            v[i] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int8)
