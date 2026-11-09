"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    # Edge lookup: for a pair (a,b), is it an edge?
    def is_edge(a, b):
        if a == b:
            return False
        return (min(a, b), max(a, b)) in edge_set

    # Check if all cross pairs between L and R are edges
    def is_biclique(L, R, remaining_edges):
        for u in L:
            for v in R:
                if u == v:
                    return False
                if not is_edge(u, v):
                    return False
        return True

    # Get the set of edges covered by biclique (L, R)
    def biclique_edges(L, R):
        es = set()
        for u in L:
            for v in R:
                if u != v:
                    es.add((min(u, v), max(u, v)))
        return es

    # Greedy: find largest biclique from remaining edges
    def find_largest_biclique(remaining_edges, adj_remaining):
        """Find a large biclique whose edges are all in remaining_edges."""
        if not remaining_edges:
            return None, None
        
        best_L = None
        best_R = None
        best_size = 0
        
        # Try each vertex as potential left anchor
        candidates = sorted(remaining_edges, key=lambda e: (-len(adj_remaining[e[0]]), -len(adj_remaining[e[1]]), e[0], e[1]))
        
        for (u, v) in candidates[:50]:  # limit candidates for speed
            if time.time() > deadline:
                break
            # Try biclique with u on left, v on right
            # Left side: u plus other vertices that connect to all of R
            # Right side: v plus other vertices that connect to all of L
            
            # Start with L={u}, R={v}
            L = {u}
            R = {v}
            
            # Try to expand: add vertices to L or R
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Try adding to L
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # w can join L if w connects to all in R
                    can_add = True
                    for r in R:
                        if not is_edge(w, r) or (min(w,r), max(w,r)) not in remaining_edges:
                            can_add = False
                            break
                    if can_add:
                        L.add(w)
                        changed = True
                # Try adding to R
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # w can join R if w connects to all in L
                    can_add = True
                    for l in L:
                        if not is_edge(w, l) or (min(w,l), max(w,l)) not in remaining_edges:
                            can_add = False
                            break
                    if can_add:
                        R.add(w)
                        changed = True
            
            if L and R:
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_L = set(L)
                    best_R = set(R)
        
        if best_L is None:
            # Fallback: single edge
            e = next(iter(remaining_edges))
            return {e[0]}, {e[1]}
        
        return best_L, best_R

    # Main greedy algorithm
    remaining = set(edge_set)
    adj_remaining = [set() for _ in range(n)]
    for u, v in edge_set:
        adj_remaining[u].add(v)
        adj_remaining[v].add(u)
    
    bicliques = []
    
    while remaining and time.time() < deadline:
        L, R = find_largest_biclique(remaining, adj_remaining)
        if L is None:
            # Shouldn't happen, but safety
            e = next(iter(remaining))
            L, R = {e[0]}, {e[1]}
        
        # Verify all cross edges are in remaining
        valid = True
        be = set()
        for u in L:
            for v in R:
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key not in remaining:
                    valid = False
                    break
                be.add(key)
            if not valid:
                break
        
        if not valid:
            # Fallback: just take one edge
            e = next(iter(remaining))
            L, R = {e[0]}, {e[1]}
            be = {e}
        
        bicliques.append([sorted(L), sorted(R)])
        
        # Remove covered edges
        for e in be:
            remaining.discard(e)
            u, v = e
            adj_remaining[u].discard(v)
            adj_remaining[v].discard(u)
    
    # If any edges remain (shouldn't), add them as single-edge bicliques
    for e in remaining:
        bicliques.append([list(e), [0]])  # placeholder, shouldn't happen
    
    # Merge step: try to merge bicliques that are compatible
    # Two bicliques (L1,R1) and (L2,R2) can be merged if:
    # - L1 and L2 are disjoint, R1 and R2 are disjoint (or we can rearrange)
    # - All cross edges between (L1∪L2) and (R1∪R2) that were in the original graph
    #   are covered by exactly one of the two bicliques
    # Actually, simpler: merge (L1,R1) and (L2,R2) into (L1∪L2, R1∪R2) 
    # if all edges in the new biclique are in the union of their edge sets
    
    def try_merge(bicliques):
        merged = True
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_bicliques = []
            used = [False] * len(bicliques)
            for i in range(len(bicliques)):
                if used[i]:
                    continue
                L1, R1 = bicliques[i][0], bicliques[i][1]
                for j in range(i+1, len(bicliques)):
                    if used[j]:
                        continue
                    L2, R2 = bicliques[j][0], bicliques[j][1]
                    # Try merging: (L1∪L2, R1∪R2)
                    Lnew = set(L1) | set(L2)
                    Rnew = set(R1) | set(R2)
                    if not Lnew or not Rnew:
                        continue
                    # Check: all cross edges must be in edge_set
                    ok = True
                    for u in Lnew:
                        for v in Rnew:
                            if u == v:
                                ok = False
                                break
                            if (min(u,v), max(u,v)) not in edge_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        # Check: the union of edges from biclique i and j equals edges of new biclique
                        edges_i = set()
                        for u in L1:
                            for v in R1:
                                if u != v:
                                    edges_i.add((min(u,v), max(u,v)))
                        edges_j = set()
                        for u in L2:
                            for v in R2:
                                if u != v:
                                    edges_j.add((min(u,v), max(u,v)))
                        edges_new = set()
                        for u in Lnew:
                            for v in Rnew:
                                if u != v:
                                    edges_new.add((min(u,v), max(u,v)))
                        if edges_i | edges_j == edges_new:
                            # Merge!
                            new_bicliques.append([sorted(Lnew), sorted(Rnew)])
                            used[i] = True
                            used[j] = True
                            changed = True
                            break
                if not used[i]:
                    new_bicliques.append(bicliques[i])
            bicliques = new_bicliques
            merged = True
        return bicliques
    
    bicliques = try_merge(bicliques)
    
    # Final safety check and convert to proper types
    result = []
    for L, R in bicliques:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        if L and R:
            result.append([L, R])
    
    # Verify the result
    covered = set()
    for L, R in result:
        for u in L:
            for v in R:
                if u != v:
                    key = (min(u, v), max(u, v))
                    if key in covered:
                        # Duplicate edge - invalid
                        # Fallback to star decomposition
                        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(P['n']) if any(a == u for a, v in P['edges'])]
                    covered.add(key)
    
    if covered != edge_set:
        # Missing edges - fallback
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(P['n']) if any(a == u for a, v in P['edges'])]
    
    return result
# EVOLVE-BLOCK-END
