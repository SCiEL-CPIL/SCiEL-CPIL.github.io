import numpy as np
import time
import itertools
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = 145.0
    
    D = 11
    
    # Generate W3 and W4 vectors
    vectors = []
    # W3
    for s in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
            
    # W4
    for s in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int8)
    n = len(V)
    
    # Build conflict graph
    # To speed up, we can use the slot logic or direct dot products.
    # n=6600. n^2 is 43M. Too slow in Python loops.
    # Use NumPy for blocks?
    
    adj = [[] for _ in range(n)]
    
    # We can optimize by grouping by support type?
    # Or just compute dot products in batches.
    
    # Batch 1: W3-W3
    # Batch 2: W3-W4
    # Batch 3: W4-W4
    
    # Indices
    n_w3 = 1320
    n_w4 = 5280
    
    # W3-W3
    V3 = V[:n_w3]
    d33 = V3 @ V3.T
    # Conflict if d > 1
    indices = np.argwhere(d33 > 1)
    for i, j in indices:
        if i != j:
            adj[i].append(j)
            adj[j].append(i)
            
    # W4-W4
    V4 = V[n_w3:]
    d44 = V4 @ V4.T
    # Conflict if d > 2
    indices = np.argwhere(d44 > 2)
    for i, j in indices:
        if i != j:
            idx1 = n_w3 + i
            idx2 = n_w3 + j
            adj[idx1].append(idx2)
            adj[idx2].append(idx1)
            
    # W3-W4
    d34 = V3 @ V4.T
    # Conflict if d > 1
    indices = np.argwhere(d34 > 1)
    for i, j in indices:
        idx1 = i
        idx2 = n_w3 + j
        adj[idx1].append(idx2)
        adj[idx2].append(idx1)
        
    initial_deg = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    best_k = 0
    best_selected = None
    
    max_restarts = 200 # Reduced to allow larger n
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        avail = np.ones(n, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n, dtype=bool)
        
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
        
        # Local search
        conflict_count = np.zeros(n, dtype=np.int32)
        sel_indices = np.where(selected)[0]
        for s in sel_indices:
            for v in adj[s]:
                if not selected[v]:
                    conflict_count[v] += 1
        
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 15:
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
            
        # Swaps
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            continue
            
        for _ in range(20):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0: break
            
            rem = sel_indices[rng.integers(len(sel_indices))]
            selected[rem] = False
            for v in adj[rem]:
                if not selected[v]:
                    conflict_count[v] -= 1
            
            improved = True
            while improved:
                improved = False
                elapsed = time.time() - start_time
                if elapsed > deadline - 5: break
                
                addable = np.where((conflict_count == 0) & (~selected))[0]
                if len(addable) == 0: break
                
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
        
    result = V[best_selected]
    return result.astype(np.int64)
