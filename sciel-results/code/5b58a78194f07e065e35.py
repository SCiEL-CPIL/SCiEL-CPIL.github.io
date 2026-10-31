import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0
    
    D = 11
    
    # Generate all weight-4 vectors (norm 2)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280
    
    # Build conflict graph using support-based optimization
    # Two weight-4 vectors conflict if dot > 2
    # If supports share <= 2 elements, dot <= 2 automatically (no conflict)
    # Only need to check pairs whose supports share >= 3 elements
    
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    adj = [[] for _ in range(n_total)]
    
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]
                
                base1 = s1 * 16
                base2 = s2 * 16
                
                v1 = all_vectors_np[base1:base1+16]
                v2 = all_vectors_np[base2:base2+16]
                
                d = v1 @ v2.T
                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    adj[idx1].append(idx2)
                    adj[idx2].append(idx1)
    
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    best_k = 0
    best_selected = None
    
    max_restarts = 300
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        # Greedy Minimum Degree Independent Set
        while avail.any():
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
            
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            min_rel = np.where(degs == min_deg)[0]
            pick_rel = min_rel[rng.integers(len(min_rel))]
            u = avail_indices[pick_rel]
            
            selected[u] = True
            avail[u] = False
            
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        k = np.sum(selected)
        
        # Local search: add non-conflicting vectors
        conflict_count = np.zeros(n_total, dtype=np.int32)
        sel_indices = np.where(selected)[0]
        for s in sel_indices:
            for v in adj[s]:
                if not selected[v]:
                    conflict_count[v] += 1
        
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            addable = np.where((conflict_count == 0) & (~selected))[0]
            if len(addable) == 0:
                break
            
            for idx in addable:
                selected[idx] = True
                conflict_count[idx] = 0
                for v in adj[idx]:
                    if not selected[v]:
                        conflict_count[v] += 1
                improved = True
        
        k = np.sum(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
        
        # Swap-based local search
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            continue
        
        for _ in range(30):
            elapsed = time.time() - start_time
            if elapsed > deadline - 3:
                break
            
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                break
            
            rem = sel_indices[rng.integers(len(sel_indices))]
            selected[rem] = False
            for v in adj[rem]:
                if not selected[v]:
                    conflict_count[v] -= 1
            
            improved = True
            while improved:
                improved = False
                elapsed = time.time() - start_time
                if elapsed > deadline - 3:
                    break
                
                addable = np.where((conflict_count == 0) & (~selected))[0]
                if len(addable) == 0:
                    break
                
                for idx in addable:
                    selected[idx] = True
                    conflict_count[idx] = 0
                    for v in adj[idx]:
                        if not selected[v]:
                            conflict_count[v] += 1
                    improved = True
            
            k_new = np.sum(selected)
            if k_new > k:
                k = k_new
                if k > best_k:
                    best_k = k
                    best_selected = selected.copy()
                break
            else:
                selected[rem] = True
                for v in adj[rem]:
                    if not selected[v]:
                        conflict_count[v] += 1
    
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
