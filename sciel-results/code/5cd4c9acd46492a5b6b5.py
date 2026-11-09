"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 82.0  # safety margin before 90s kill
    
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Track uncovered edges
    uncovered = set()
    for u, v in edges:
        uncovered.add((min(u,v), max(u,v)))
    
    if not uncovered:
        return []
    
    def find_maximal_biclique_in_graph(u, v, edge_set):
        """Find maximal biclique containing edge (u,v) in subgraph defined by edge_set."""
        L = {u}
        R = {v}
        
        changed = True
        while changed:
            changed = False
            
            # Expand L: add vertices connected to ALL in R via edges in edge_set
            candidates_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                all_connected = True
                for r in R:
                    a, b = min(w, r), max(w, r)
                    if (a, b) not in edge_set:
                        all_connected = False
                        break
                if all_connected:
                    candidates_L.add(w)
            
            if candidates_L:
                L = L | candidates_L
                changed = True
            
            # Expand R: add vertices connected to ALL in L via edges in edge_set
            candidates_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                all_connected = True
                for l in L:
                    a, b = min(w, l), max(w, l)
                    if (a, b) not in edge_set:
                        all_connected = False
                        break
                if all_connected:
                    candidates_R.add(w)
            
            if candidates_R:
                R = R | candidates_R
                changed = True
        
        # Ensure disjoint
        R = R - L
        L = L - R
        
        return L, R
    
    def greedy_partition(uncov, edge_order='max_biclique'):
        """Greedy biclique partition using given edge ordering strategy."""
        uncov = set(uncov)
        bicliques = []
        
        while uncov:
            if time.time() > deadline:
                for e in list(uncov):
                    u, v = e
                    bicliques.append([[u], [v]])
                break
            
            # Pick next edge based on strategy
            if edge_order == 'max_biclique':
                # Pick edge that yields largest biclique
                best_edge = None
                best_size = -1
                for e in list(uncov)[:50]:  # sample for speed
                    u, v = e
                    L, R = find_maximal_biclique_in_graph(u, v, uncov)
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_edge = e
                if best_edge is None:
                    best_edge = next(iter(uncov))
            elif edge_order == 'degree':
                best_edge = None
                best_score = -1
                for e in uncov:
                    u, v = e
                    score = len(adj[u]) + len(adj[v])
                    if score > best_score:
                        best_score = score
                        best_edge = e
            elif edge_order == 'common_neighbors':
                best_edge = None
                best_score = -1
                for e in uncov:
                    u, v = e
                    common = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        a1, b1 = min(u, w), max(u, w)
                        a2, b2 = min(v, w), max(v, w)
                        if (a1, b1) in uncov and (a2, b2) in uncov:
                            common += 1
                    score = common
                    if score > best_score:
                        best_score = score
                        best_edge = e
            elif edge_order == 'bottleneck':
                best_edge = None
                best_score = float('inf')
                for e in uncov:
                    u, v = e
                    score = min(len(adj[u]), len(adj[v]))
                    if score < best_score:
                        best_score = score
                        best_edge = e
            else:
                best_edge = next(iter(uncov))
            
            u, v = best_edge
            
            # Find maximal biclique in uncovered subgraph
            L, R = find_maximal_biclique_in_graph(u, v, uncov)
            
            if not L or not R:
                bicliques.append([[u], [v]])
                uncov.discard((min(u,v), max(u,v)))
                continue
            
            bicliques.append([list(L), list(R)])
            
            # Remove all edges in this biclique from uncovered
            for l in L:
                for r in R:
                    a, b = min(l, r), max(l, r)
                    uncov.discard((a, b))
        
        return bicliques
    
    def try_merge(bicliques):
        """Try to merge pairs of bicliques if their union forms a valid biclique in the full graph."""
        if len(bicliques) < 2:
            return bicliques
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            i = 0
            new_bicliques = list(bicliques)
            while i < len(new_bicliques) - 1:
                j = i + 1
                while j < len(new_bicliques):
                    L1, R1 = set(new_bicliques[i][0]), set(new_bicliques[i][1])
                    L2, R2 = set(new_bicliques[j][0]), set(new_bicliques[j][1])
                    
                    # Try all 4 orientations of merging
                    merged = False
                    for Lm, Rm in [(L1|L2, R1|R2), (L1|L2, R1&R2), (L1&R2, R1|L2), (L1|R1, R2|L2)]:
                        # Actually, proper merge: try (L1∪L2, R1∪R2), (L1∪L2, R1∩R2) won't work
                        # Standard: merge if L1∪L2 and R1∪R2 form a biclique
                        pass
                    
                    # Standard merge: check if (L1∪L2) × (R1∪R2) is a biclique in full graph
                    # This requires every l in L1∪L2 to connect to every r in R1∪R2
                    Lm = L1 | L2
                    Rm = R1 | R2
                    
                    if not Lm or not Rm or (Lm & Rm):
                        j += 1
                        continue
                    
                    # Check all cross edges exist in full graph
                    valid = True
                    for l in Lm:
                        for r in Rm:
                            a, b = min(l, r), max(l, r)
                            if (a, b) not in adj[l] and (a, b) not in adj[r]:
                                # Check if edge exists
                                if r not in adj[l]:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if valid:
                        # Verify this merged biclique covers exactly the union of edges from both
                        # Remove biclique j and replace i with merged
                        new_bicliques[i] = [list(Lm), list(Rm)]
                        new_bicliques.pop(j)
                        changed = True
                        break
                    j += 1
                
                if changed:
                    break
                i += 1
            
            bicliques = new_bicliques
        
        return bicliques
    
    # Try different strategies and keep the best
    best_result = None
    best_count = float('inf')
    
    strategies = ['max_biclique', 'degree', 'common_neighbors', 'bottleneck']
    
    for strat in strategies:
        if time.time() > deadline - 3:
            break
        result = greedy_partition(uncovered, edge_order=strat)
        # Try to merge
        result_merged = try_merge(result)
        if len(result_merged) < best_count:
            best_count = len(result_merged)
            best_result = result_merged
        elif len(result) < best_count:
            best_count = len(result)
            best_result = result
    
    # Convert to proper types
    if best_result is None:
        result = []
        for e in uncovered:
            u, v = e
            result.append([[u], [v]])
        return result
    
    final_result = []
    for bic in best_result:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        if left and right and not (set(left) & set(right)):
            final_result.append([left, right])
    
    return final_result
# EVOLVE-BLOCK-END
