import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    D = 11
    
    # Generate all weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Build adjacency: supports sharing exactly 3 elements
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    adj = [[] for _ in range(n_supports)]
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                adj[i].append(j)
                adj[j].append(i)
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    vec_support = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)
    
    # Precompute compatibility matrix
    # compatible[i, j] = True if dot(i, j) <= 2
    dots = all_vectors_np @ all_vectors_np.T
    compatible = dots <= 2
    
    # Mask for self-compatibility (diagonal is True, but we don't pick same index twice)
    # We will use a boolean array 'selected'
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Random ordering of supports
        order = rng.permutation(n_supports)
        
        selected = np.zeros(n_total, dtype=bool)
        selected_count = 0
        selected_by_support = [[] for _ in range(n_supports)]
        
        # Greedy Phase
        for s_idx in order:
            base = s_idx * 16
            candidates = list(range(base, base + 16))
            
            # Get adjacent supports that have selected vectors
            adj_with_selected = [n for n in adj[s_idx] if selected_by_support[n]]
            
            local_selected = []
            for cand in candidates:
                ok = True
                for n_idx in adj_with_selected:
                    for sel_idx in selected_by_support[n_idx]:
                        if not compatible[cand, sel_idx]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    local_selected.append(cand)
            
            for idx in local_selected:
                selected[idx] = True
                selected_count += 1
                selected_by_support[s_idx].append(idx)
        
        # Local Search / Fill Phase
        # Try to add any unselected vector
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 2:
                break
            
            # Iterate through unselected vectors
            # Optimization: Only check vectors whose support has adjacent selected vectors
            # If a support has no adjacent selected vectors, all its unselected vectors can be added
            for s_idx in range(n_supports):
                if not selected_by_support[s_idx]:
                    # No selected vectors in this support
                    # Check if any neighbor has selected vectors
                    has_adj_selected = False
                    for n_idx in adj[s_idx]:
                        if selected_by_support[n_idx]:
                            has_adj_selected = True
                            break
                    
                    if not has_adj_selected:
                        # Can add all 16
                        base = s_idx * 16
                        for i in range(base, base + 16):
                            if not selected[i]:
                                selected[i] = True
                                selected_count += 1
                                selected_by_support[s_idx].append(i)
                                improved = True
                        continue
                
                # Check unselected vectors in this support
                base = s_idx * 16
                for i in range(base, base + 16):
                    if selected[i]:
                        continue
                    
                    # Check compatibility with selected neighbors
                    ok = True
                    for n_idx in adj[s_idx]:
                        for sel_idx in selected_by_support[n_idx]:
                            if not compatible[i, sel_idx]:
                                ok = False
                                break
                        if not ok:
                            break
                    
                    if ok:
                        selected[i] = True
                        selected_count += 1
                        selected_by_support[s_idx].append(i)
                        improved = True
            
        if selected_count > best_k:
            best_k = selected_count
            best_selected = selected.copy()
            
        # If we have a good solution, try a few swaps?
        # For now, rely on restarts.
        
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
