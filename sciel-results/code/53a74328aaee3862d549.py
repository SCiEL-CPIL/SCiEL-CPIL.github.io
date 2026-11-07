import time
import random
import numpy as np
import scipy.optimize

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']
    
    # Build adjacency mask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Seed Solution (Baseline)
    best_bicliques = []
    for u in range(n):
        right_side = [v for a, v in edges if a == u]
        if right_side:
            best_bicliques.append([[u], right_side])
            
    best_count = len(best_bicliques)
    
    candidate_pool = set()

    def add_candidate(biclique):
        # Normalize: sort left, sort right
        L = tuple(sorted(biclique[0]))
        R = tuple(sorted(biclique[1]))
        candidate_pool.add((L, R))

    # Add baseline candidates
    for b in best_bicliques:
        add_candidate(b)

    def try_greedy(restarts=200):
        nonlocal best_bicliques, best_count, candidate_pool
        
        for r in range(restarts):
            if time.time() > deadline:
                return
            
            rem_adj = adj[:]
            current_bicliques = []
            edges_left = len(edges)
            
            while edges_left > 0:
                if time.time() > deadline:
                    return
                
                found_edge = False
                for _ in range(50):
                    u = random.randint(0, n-1)
                    mask = rem_adj[u]
                    if mask:
                        m = mask
                        candidates = []
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            candidates.append(v)
                            m ^= lsb
                        v = random.choice(candidates)
                        
                        if rem_adj[u] & (1 << v):
                            found_edge = True
                            break
                
                if not found_edge:
                    break
                
                L = {u}
                R = {v}
                changed = True
                for _ in range(20):
                    if not changed:
                        break
                    changed = False
                    
                    candidates_L = [w for w in range(n) if w not in L]
                    random.shuffle(candidates_L)
                    
                    for w in candidates_L:
                        mask_R = 0
                        for rv in R:
                            mask_R |= (1 << rv)
                        if (rem_adj[w] & mask_R) == mask_R:
                            L.add(w)
                            changed = True
                            
                    candidates_R = [w for w in range(n) if w not in R]
                    random.shuffle(candidates_R)
                    
                    for w in candidates_R:
                        valid = True
                        for lu in L:
                            if not (rem_adj[lu] & (1 << w)):
                                valid = False
                                break
                        if valid:
                            R.add(w)
                            changed = True

                biclique = [list(L), list(R)]
                current_bicliques.append(biclique)
                add_candidate(biclique)
                
                mask_L = 0
                for lu in L:
                    mask_L |= (1 << lu)
                mask_R = 0
                for rv in R:
                    mask_R |= (1 << rv)
                
                for lu in L:
                    rem_adj[lu] &= ~mask_R
                for rv in R:
                    rem_adj[rv] &= ~mask_L
                
                edges_left -= len(L) * len(R)
                
                if edges_left < 0:
                    break

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    return

    try_greedy()

    def local_search_refine(bicliques, max_iters=50):
        """Try to reduce biclique count by merging."""
        if time.time() > deadline:
            return bicliques
        
        bl = [list(b) for b in bicliques]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                return bl
            
            for i in range(len(bl)):
                if time.time() > deadline:
                    return bl
                L1, R1 = bl[i][0], bl[i][1]
                
                for j in range(len(bl)):
                    if i == j:
                        continue
                    if time.time() > deadline:
                        return bl
                    
                    L2, R2 = bl[j][0], bl[j][1]
                    
                    if set(L1) & set(L2) or set(R1) & set(R2):
                        continue
                    if set(L1) & set(R2) or set(L2) & set(R1):
                        continue
                    
                    valid = True
                    for u in L1:
                        for v in R1:
                            if not (adj[u] & (1 << v)):
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        for u in L1:
                            for v in R2:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid: break
                    if valid:
                        for u in L2:
                            for v in R1:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid: break
                    if valid:
                        for u in L2:
                            for v in R2:
                                if not (adj[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid: break
                    
                    if valid:
                        new_L = L1 + L2
                        new_R = R1 + R2
                        bl[i] = [new_L, new_R]
                        del bl[j]
                        improved = True
                        break
                
                if improved:
                    break
        
        return bl

    # Phase 2: Selection from Candidate Pool
    if candidate_pool and time.time() < deadline - 10:
        cands = list(candidate_pool)
        n_cands = len(cands)
        n_edges = len(edges)
        
        # Map edges to indices
        edge_index = {}
        for i, (u, v) in enumerate(edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        # Build A matrix (n_edges x n_cands)
        A = np.zeros((n_edges, n_cands), dtype=np.int8)
        for c_idx, (L, R) in enumerate(cands):
            for u in L:
                for v in R:
                    # Check if edge exists in original graph
                    if (u, v) in edge_index:
                        A[edge_index[(u, v)], c_idx] = 1
                    elif (v, u) in edge_index:
                        A[edge_index[(v, u)], c_idx] = 1

        selected = None
        
        # Try MILP
        try:
            c = np.ones(n_cands)
            constraints = [scipy.optimize.LinearConstraint(A, 1, np.inf)]
            integrality = np.ones(n_cands)
            bounds = [(0, 1)] * n_cands
            
            res = scipy.optimize.milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                x = res.x
                selected_indices = [i for i in range(n_cands) if x[i] > 0.5]
                selected = [cands[i] for i in selected_indices]
        except Exception:
            pass

        # Fallback: Greedy Set Cover
        if selected is None:
            selected = []
            covered = np.zeros(n_edges, dtype=bool)
            remaining_cands = list(range(n_cands))
            
            while not np.all(covered):
                if time.time() > deadline - 5:
                    break
                best_idx = -1
                best_gain = -1
                for idx in remaining_cands:
                    gain = np.sum(A[:, idx] & ~covered)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx == -1 or best_gain == 0:
                    break
                selected.append(cands[best_idx])
                covered |= A[:, best_idx]
                remaining_cands.remove(best_idx)
                
        if selected:
            # Convert to list of lists
            sel_bicliques = [[list(L), list(R)] for L, R in selected]
            if len(sel_bicliques) < best_count:
                best_bicliques = sel_bicliques
                best_count = len(best_bicliques)

    # Phase 3: Refine
    if best_bicliques:
        refined = local_search_refine(best_bicliques, max_iters=100)
        if len(refined) < best_count:
            best_count = len(refined)
            best_bicliques = refined

    # Fallback
    if not best_bicliques:
        for u in range(n):
            right_side = [v for a, v in edges if a == u]
            if right_side:
                best_bicliques.append([[u], right_side])

    # Convert to required format
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
