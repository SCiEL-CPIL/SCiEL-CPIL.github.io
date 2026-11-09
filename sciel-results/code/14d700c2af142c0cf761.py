import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    # Build adjacency sets for original graph
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Normalize edges to (min, max)
    norm_edges = set()
    for u, v in edges:
        if u < v:
            norm_edges.add((u, v))
        else:
            norm_edges.add((v, u))
            
    all_edges = list(norm_edges)
    
    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Strategies for edge selection
    # 1. Degree sum
    # 2. Common neighbors
    # 3. Random
    
    strategies = ['deg_sum', 'common', 'random']
    
    for strat in strategies:
        if time.time() > deadline:
            break
            
        if strat == 'deg_sum':
            # Sort edges by degree sum of endpoints
            edge_weights = []
            for u, v in all_edges:
                edge_weights.append((-(len(adj[u]) + len(adj[v])), (u, v)))
            edge_weights.sort()
            ordered_edges = [e for w, e in edge_weights]
        elif strat == 'common':
            edge_weights = []
            for u, v in all_edges:
                common = len(adj[u].intersection(adj[v]))
                edge_weights.append((-common, (u, v)))
            edge_weights.sort()
            ordered_edges = [e for w, e in edge_weights]
        else: # random
            ordered_edges = all_edges[:]
            random.shuffle(ordered_edges)
            
        # Greedy algorithm
        uncovered = set(all_edges)
        bicliques = []
        
        # We iterate through the ordered edges.
        # If an edge is still uncovered, we grow a biclique from it.
        for u, v in ordered_edges:
            if (u, v) not in uncovered:
                continue
                
            # Grow biclique
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Try to add to L
                candidates_L = [w for w in range(n) if w not in L and w not in R]
                for w in candidates_L:
                    # w can be in L if all r in R are neighbors of w
                    if adj[w].issuperset(R):
                        L.add(w)
                        changed = True
                        # Note: Adding to L might allow more to be added to R, or vice versa.
                        # We restart the loop to be safe and simple.
                        break 
                if changed:
                    continue
                # Try to add to R
                candidates_R = [w for w in range(n) if w not in L and w not in R]
                for w in candidates_R:
                    # w can be in R if all l in L are neighbors of w
                    if adj[w].issuperset(L):
                        R.add(w)
                        changed = True
                        break
            if changed:
                continue
                
            # The maximal biclique is (L, R)
            # But we must ensure we only cover uncovered edges?
            # The biclique (L, R) covers all pairs (l, r) with l in L, r in R.
            # Some of these might already be covered?
            # No, we are building from an uncovered edge.
            # However, if we added vertices, we might cover edges that are NOT in `uncovered`?
            # The condition "Every cross pair must be a host edge" is satisfied by construction.
            # The condition "cross-edge sets must partition all pinned graph edges exactly once" means we cannot cover an edge that was already covered by a previous biclique.
            # So, we should only accept the biclique if all its edges are in `uncovered`.
            # If some edges in (L, R) are already covered, we cannot use this (L, R) directly.
            # We must shrink L or R to avoid covering already covered edges.
            
            # Let's check if (L, R) is valid w.r.t uncovered edges.
            # Valid if for all l in L, r in R, (l, r) in uncovered.
            # If not, we need to reduce L or R.
            # Simplest: keep only the parts that are fully uncovered?
            # Actually, if (l1, r1) is covered, we can't include both l1 and r1.
            # We can try to find a maximal subset L' of L and R' of R such that all (l, r) in L' x R' are in uncovered.
            # This is finding a maximal biclique in the subgraph induced by uncovered edges on L U R?
            # No, just L x R restricted to uncovered.
            
            # Let's refine L and R.
            # Iterate and remove vertices that have any "conflict" (edge not in uncovered).
            valid = True
            while valid:
                valid = False
                for l in list(L):
                    for r in R:
                        edge = (min(l, r), max(l, r))
                        if edge not in uncovered:
                            # Conflict
                            L.remove(l)
                            valid = True
                            break
                    if valid:
                        break
                if valid:
                    continue
                for r in list(R):
                    for l in L:
                        edge = (min(l, r), max(l, r))
                        if edge not in uncovered:
                            R.remove(r)
                            valid = True
                            break
                    if valid:
                        break
                            
            if not L or not R:
                continue
                
            # Add biclique
            bicliques.append((list(L), list(R)))
            
            # Remove covered edges from uncovered
            for l in L:
                for r in R:
                    edge = (min(l, r), max(l, r))
                    if edge in uncovered:
                        uncovered.remove(edge)
                        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
        # We can also try random restarts for the 'random' strategy
        for i in range(5):
            if time.time() > deadline:
                break
            ordered_edges = all_edges[:]
            random.shuffle(ordered_edges)
            uncovered = set(all_edges)
            bicliques = []
            for u, v in ordered_edges:
                if (u, v) not in uncovered:
                    continue
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    candidates_L = [w for w in range(n) if w not in L and w not in R]
                    random.shuffle(candidates_L)
                    for w in candidates_L:
                        if adj[w].issuperset(R):
                            L.add(w)
                            changed = True
                            break
                    if changed:
                        continue
                    candidates_R = [w for w in range(n) if w not in L and w not in R]
                    random.shuffle(candidates_R)
                    for w in candidates_R:
                        if adj[w].issuperset(L):
                            R.add(w)
                            changed = True
                            break
                if changed:
                    continue
                    
                valid = True
                while valid:
                    valid = False
                    for l in list(L):
                        for r in R:
                            edge = (min(l, r), max(l, r))
                            if edge not in uncovered:
                                L.remove(l)
                                valid = True
                                break
                        if valid:
                            break
                    if valid:
                        continue
                    for r in list(R):
                        for l in L:
                            edge = (min(l, r), max(l, r))
                            if edge not in uncovered:
                                R.remove(r)
                                valid = True
                                break
                        if valid:
                            break
                if not L or not R:
                    continue
                bicliques.append((list(L), list(R)))
                for l in L:
                    for r in R:
                        edge = (min(l, r), max(l, r))
                        if edge in uncovered:
                            uncovered.remove(edge)
                            
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques

    # Convert to required format
    # "list of bicliques, each [left_vertices,right_vertices]"
    # Types: list or tuple. Integers must be Python int.
    if best_bicliques is None:
        # Fallback to seed
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
        
    result = []
    for L, R in best_bicliques:
        result.append([list(L), list(R)])
        
    return result
