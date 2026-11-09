import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    num_edges = len(edges)
    
    # Build original adjacency masks
    adj_orig = [0] * n
    for u, v in edges:
        adj_orig[u] |= (1 << v)
        adj_orig[v] |= (1 << u)
    
    # Edge index mapping
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    best_result = None
    best_count = float('inf')
    
    # Multi-start randomized greedy + local search
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random seed for this restart
        rng = random.Random(restart * 12345 + 67890)
        
        # Phase 1: Randomized greedy biclique construction
        remaining_edges = list(edges)
        rng.shuffle(remaining_edges)
        
        # Track coverage with a set of covered edge tuples
        covered = set()
        bicliques = []
        
        # For each restart, try different vertex orderings
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        for start_v in vertices:
            if time.time() > deadline:
                break
            
            # Find all remaining edges incident to start_v
            incident = []
            for i, (u, v) in enumerate(remaining_edges):
                if (u, v) in covered:
                    continue
                if u == start_v:
                    incident.append((i, v))
                elif v == start_v:
                    incident.append((i, u))
            
            if not incident:
                continue
            
            # Build initial biclique centered on start_v
            L = [start_v]
            R = []
            R_mask = 0
            used_edges = set()
            
            for edge_idx, other in incident:
                u, v = remaining_edges[edge_idx]
                if u == start_v:
                    other = v
                else:
                    other = u
                
                # Check if other can be added to R (connected to all in L)
                can_add = True
                for lv in L:
                    if not (adj_orig[lv] & (1 << other)):
                        can_add = False
                        break
                
                if can_add:
                    R.append(other)
                    R_mask |= (1 << other)
                    # Mark edges from L to other as used
                    for lv in L:
                        a, b = (lv, other) if lv < other else (other, lv)
                        used_edges.add((a, b))
            
            if not R:
                continue
            
            # Expand biclique greedily
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                
                # Try adding to L
                candidates_L = []
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # w must be connected to all in R
                    if (adj_orig[w] & R_mask) != R_mask:
                        continue
                    # Check if adding w covers new edges
                    new_edges = 0
                    for rv in R:
                        a, b = (w, rv) if w < rv else (rv, w)
                        if (a, b) not in used_edges:
                            new_edges += 1
                    if new_edges > 0:
                        candidates_L.append((new_edges, w))
                
                # Try adding to R
                candidates_R = []
                L_mask = 0
                for lv in L:
                    L_mask |= (1 << lv)
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # w must be connected to all in L
                    if (adj_orig[w] & L_mask) != L_mask:
                        continue
                    new_edges = 0
                    for lv in L:
                        a, b = (lv, w) if lv < w else (w, lv)
                        if (a, b) not in used_edges:
                            new_edges += 1
                    if new_edges > 0:
                        candidates_R.append((new_edges, w))
                
                best_gain = 0
                best_action = None
                
                for gain, w in candidates_L:
                    if gain > best_gain:
                        best_gain = gain
                        best_action = ('L', w)
                
                for gain, w in candidates_R:
                    if gain > best_gain:
                        best_gain = gain
                        best_action = ('R', w)
                
                if best_action:
                    side, w = best_action
                    if side == 'L':
                        L.append(w)
                        for rv in R:
                            a, b = (w, rv) if w < rv else (rv, w)
                            used_edges.add((a, b))
                    else:
                        R.append(w)
                        for lv in L:
                            a, b = (lv, w) if lv < w else (w, lv)
                            used_edges.add((a, b))
                    changed = True
            
            # Add biclique if it covers edges
            if used_edges:
                bicliques.append([sorted(L), sorted(R), used_edges])
                covered.update(used_edges)
        
        # Check if all edges covered
        if len(covered) < num_edges:
            # Cover remaining edges one by one
            for u, v in edges:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in covered:
                    bicliques.append([[a], [b], {(a, b)}])
                    covered.add((a, b))
        
        # Phase 2: Local search - try to merge bicliques
        # Represent bicliques as [L_list, R_list, edge_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(bicliques)):
                    Li, Ri, Ei = bicliques[i]
                    Lj, Rj, Ej = bicliques[j]
                    
                    # Check if union is a valid biclique
                    # L_new = Li ∪ Lj, R_new = Ri ∪ Rj
                    # Need: all pairs in L_new × R_new are edges in original graph
                    valid = True
                    for u in Li:
                        for v in Rj:
                            if not (adj_orig[u] & (1 << v)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for u in Lj:
                            for v in Ri:
                                if not (adj_orig[u] & (1 << v)):
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        # Merge
                        L_new = sorted(set(Li) | set(Lj))
                        R_new = sorted(set(Ri) | set(Rj))
                        E_new = Ei | Ej
                        bicliques[i] = [L_new, R_new, E_new]
                        bicliques.pop(j)
                        improved = True
                        break
            
            # Try moving edges between bicliques to enable merges
            if not improved and len(bicliques) > 1 and time.time() < deadline:
                moved = False
                for i in range(len(bicliques)):
                    if time.time() > deadline:
                        break
                    Li, Ri, Ei = bicliques[i]
                    # Try moving one edge from biclique i to see if it enables a merge
                    # This is complex; skip for now and rely on multi-start
                    pass
        
        # Count bicliques
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_result = bicliques[:]
        
        # Early stop if we found a very good solution
        if best_count <= 5:
            break
    
    # If no valid result found, fall back to trivial solution
    if best_result is None:
        best_result = []
        for u, v in edges:
            best_result.append([[u], [v], {(u, v) if u < v else (v, u)}])
    
    # Convert to required format
    final_result = []
    for bic in best_result:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
    
    return final_result
