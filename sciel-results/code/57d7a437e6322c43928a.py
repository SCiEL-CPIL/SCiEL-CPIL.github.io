"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Strategy: Use structured construction based on geometric symmetries
    # Focus on weight-2 vectors (norm^2=2) and carefully add weight-4 vectors (norm^2=4)
    # Condition: min pairwise dist^2 >= max norm^2 = 4
    # For weight-2 vectors: dist^2 = 2+2-2*dot = 4-2*dot, need dot <= 0
    # For weight-2 and weight-4: dist^2 = 2+4-2*dot = 6-2*dot, need dot <= 1
    # For weight-4 vectors: dist^2 = 4+4-2*dot = 8-2*dot, need dot <= 2
    
    # Generate all weight-2 vectors
    w2_candidates = []
    w2_indices = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            w2_candidates.append(v)
            w2_indices.append(positions)
    
    w2_candidates = np.array(w2_candidates, dtype=np.int64)
    n_w2 = len(w2_candidates)
    
    # Generate all weight-4 vectors
    w4_candidates = []
    w4_indices = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            w4_candidates.append(v)
            w4_indices.append(positions)
    
    w4_candidates = np.array(w4_candidates, dtype=np.int64)
    n_w4 = len(w4_candidates)
    
    # Compute conflicts among weight-2 vectors
    # dist^2 = 4 - 2*dot, conflict if dot > 0
    dots_w2 = np.dot(w2_candidates, w2_candidates.T)
    conflicts_w2 = dots_w2 > 0
    np.fill_diagonal(conflicts_w2, False)
    
    # Compute conflicts between weight-2 and weight-4
    # dist^2 = 6 - 2*dot, conflict if dot > 1
    dots_w2_w4 = np.dot(w2_candidates, w4_candidates.T)
    conflicts_w2_w4 = dots_w2_w4 > 1
    
    # Compute conflicts among weight-4 vectors
    # dist^2 = 8 - 2*dot, conflict if dot > 2
    dots_w4 = np.dot(w4_candidates, w4_candidates.T)
    conflicts_w4 = dots_w4 > 2
    np.fill_diagonal(conflicts_w4, False)
    
    # Build neighbor lists for efficient checking
    # For weight-2: neighbor if conflict
    w2_neighbors = [set() for _ in range(n_w2)]
    for i in range(n_w2):
        w2_neighbors[i] = set(np.where(conflicts_w2[i])[0].tolist())
    
    # For weight-4: neighbor in w4 if conflict
    w4_neighbors_w4 = [set() for _ in range(n_w4)]
    for i in range(n_w4):
        w4_neighbors_w4[i] = set(np.where(conflicts_w4[i])[0].tolist())
    
    # For weight-4: which weight-2 vectors conflict with it
    w4_neighbors_w2 = [set() for _ in range(n_w4)]
    for j in range(n_w4):
        w4_neighbors_w2[j] = set(np.where(conflicts_w2_w4[:, j])[0].tolist())
    
    def structured_greedy_w2():
        """
        Structured greedy for weight-2 vectors using coordinate-based ordering.
        Process coordinates in pairs, use symmetry to maximize packing.
        """
        selected = set()
        
        # For weight-2 vectors, two vectors conflict if they share exactly one coordinate
        # with the same sign, or share both coordinates with opposite signs (dot = -1, dist^2=6, OK)
        # Actually: dot = sum of products. Two weight-2 vectors:
        # - Disjoint support: dot=0, dist^2=4, OK
        # - Share one coordinate: dot = ±1, if same sign dot=1 (conflict), if opposite dot=-1 (OK)
        # - Same support: dot = ±2, if same dot=2 (conflict), if opposite dot=-2 (OK)
        
        # Key insight: we can partition by support. For each pair of coordinates,
        # we can have at most 2 vectors: (a,b) and (-a,-b) are OK (dot=-2),
        # but (a,b) and (a,-b) conflict (dot=0? no wait...)
        
        # Let me recompute: v1=(a,b,0,...), v2=(c,d,0,...) on same coords {i,j}
        # dot = a*c + b*d
        # (1,1) and (1,1): dot=2, conflict
        # (1,1) and (1,-1): dot=0, dist^2=4, OK
        # (1,1) and (-1,1): dot=0, OK
        # (1,1) and (-1,-1): dot=-2, dist^2=8, OK
        # So on same support, we can have up to 4 vectors? No:
        # (1,1) conflicts with (1,1) only (itself)
        # (1,1) and (1,-1): dot=0, OK
        # (1,1) and (-1,1): dot=0, OK  
        # (1,1) and (-1,-1): dot=-2, OK
        # (1,-1) and (-1,1): dot=0, OK
        # (1,-1) and (-1,-1): dot=0, OK
        # (-1,1) and (-1,-1): dot=0, OK
        # So all 4 sign combinations on same support are pairwise compatible!
        
        # But vectors on different supports can conflict if they share one coordinate with same sign.
        
        # Strategy: For each coordinate pair, select all 4 sign variants.
        # Then check cross-pair conflicts and remove as needed.
        
        # Actually, let's use a simpler greedy: process vectors in a smart order.
        # Order by: for each coordinate pair, process all 4 sign variants together.
        
        # Build ordering: group by support
        support_to_vectors = {}
        for i in range(n_w2):
            supp = tuple(sorted(w2_indices[i]))
            if supp not in support_to_vectors:
                support_to_vectors[supp] = []
            support_to_vectors[supp].append(i)
        
        # For each support, we want to select a subset
        # Start with all 4 per support, then resolve conflicts
        
        # Greedy: for each support in order, try to add all 4, remove conflicting ones
        selected = set()
        
        # Order supports by some heuristic - maybe by number of already-selected neighbors
        supports_list = list(support_to_vectors.keys())
        
        # Process in random order with multiple restarts
        best_selected = set()
        
        n_restarts = 20
        for trial in range(n_restarts):
            if time.time() > deadline - 30.0:
                break
            
            random.seed(1000 + trial)
            random.shuffle(supports_list)
            
            cur_selected = set()
            
            for supp in supports_list:
                vecs = support_to_vectors[supp]
                # Try to add all 4 vectors of this support
                for v in vecs:
                    if v in cur_selected:
                        continue
                    # Check if v conflicts with any selected vector
                    compatible = True
                    for s in cur_selected:
                        if s in w2_neighbors[v]:
                            compatible = False
                            break
                    if compatible:
                        cur_selected.add(v)
            
            if len(cur_selected) > len(best_selected):
                best_selected = cur_selected.copy()
        
        return best_selected
    
    def add_w4_to_w2(w2_sel, max_iterations=500):
        """
        Given a set of selected weight-2 vectors, add compatible weight-4 vectors.
        """
        w4_selected = set()
        
        # For each weight-4 vector, check if it's compatible with all selected w2 vectors
        # A w4 vector j is compatible with w2 vector i if dot(w2_i, w4_j) <= 1
        
        # Precompute: for each w4, which w2 vectors does it conflict with
        # Then a w4 is addable if none of its conflicting w2 are selected
        
        # Greedy: sort w4 by number of conflicts with selected w2, pick min first
        w4_conflict_count = []
        for j in range(n_w4):
            # Count how many selected w2 conflict with this w4
            count = len(w4_neighbors_w2[j] & w2_sel)
            w4_conflict_count.append((count, j))
        
        w4_conflict_count.sort()
        
        for count, j in w4_conflict_count:
            if time.time() > deadline - 20.0:
                break
            if count > 0:
                continue
            # Check compatibility with already selected w4
            compatible_w4 = True
            for k in w4_selected:
                if k in w4_neighbors_w4[j]:
                    compatible_w4 = False
                    break
            if compatible_w4:
                w4_selected.add(j)
        
        return w4_selected
    
    def local_search_2opt(w2_sel, w4_sel):
        """
        2-opt style local search: try swapping one w2 for another, or one w4 for another.
        """
        w2_sel = set(w2_sel)
        w4_sel = set(w4_sel)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try to add any w2 that fits
            for v in range(n_w2):
                if time.time() > deadline - 10.0:
                    break
                if v not in w2_sel:
                    if not (w2_neighbors[v] & w2_sel):
                        w2_sel.add(v)
                        improved = True
            
            # Try to add any w4 that fits
            for j in range(n_w4):
                if time.time() > deadline - 10.0:
                    break
                if j not in w4_sel:
                    if not (w4_neighbors_w2[j] & w2_sel) and not (w4_neighbors_w4[j] & w4_sel):
                        w4_sel.add(j)
                        improved = True
            
            # Try 1-for-1 swaps on w2
            if not improved:
                for v in list(w2_sel):
                    if time.time() > deadline - 10.0:
                        break
                    # Try removing v and adding a different w2
                    test_set = w2_sel - {v}
                    # Find w2 vectors that are compatible with test_set but not in test_set
                    for u in range(n_w2):
                        if u in test_set or u == v:
                            continue
                        if not (w2_neighbors[u] & test_set):
                            # Check if u is better than v (has fewer conflicts with remaining)
                            # For now, just accept if it allows more w4
                            new_w4 = add_w4_to_w2(test_set, max_iterations=100)
                            old_w4 = add_w4_to_w2(w2_sel, max_iterations=100)
                            if len(new_w4) > len(old_w4):
                                w2_sel = test_set.union({u})
                                w4_sel = new_w4
                                improved = True
                                break
        
        return w2_sel, w4_sel
    
    # Phase 1: Structured greedy for weight-2
    w2_best = structured_greedy_w2()
    
    # Phase 2: Add weight-4 vectors
    w4_best = add_w4_to_w2(w2_best)
    
    # Phase 3: Local search
    w2_final, w4_final = local_search_2opt(w2_best, w4_best)
    
    # Build final result
    selected_vectors = []
    for v in w2_final:
        selected_vectors.append(w2_candidates[v])
    for j in w4_final:
        selected_vectors.append(w4_candidates[j])
    
    if not selected_vectors:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    result = np.array(selected_vectors, dtype=np.int64)
    
    # Verify and return
    return result

# EVOLVE-BLOCK-END
