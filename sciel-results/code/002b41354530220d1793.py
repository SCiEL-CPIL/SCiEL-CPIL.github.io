"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_biclique(left, right, adj):
    """Check if left x right is a biclique in the graph."""
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True

def _biclique_edges(left, right, adj):
    """Get the set of edges covered by this biclique."""
    edges = set()
    for u in left:
        for v in right:
            if v in adj[u]:
                edges.add((min(u,v), max(u,v)))
    return edges

def _find_max_biclique(uncovered, adj, deadline):
    """Find a large biclique using greedy approach on uncovered edges."""
    if not uncovered:
        return None
    
    # Build bipartite structure from uncovered edges
    # Try to find a large complete bipartite subgraph
    best_left = set()
    best_right = set()
    best_count = 0
    
    # Get all vertices involved in uncovered edges
    involved = set()
    for e in uncovered:
        involved.add(e[0])
        involved.add(e[1])
    
    # Greedy: try each vertex as potential left seed, expand
    for seed in sorted(involved):
        if time.time() > deadline:
            break
        # Start with seed on one side, try to find compatible right side
        # Try seed on left
        left_candidates = {seed}
        # Right side: vertices connected to all left candidates via uncovered edges
        right_candidates = set()
        for e in uncovered:
            if e[0] == seed:
                right_candidates.add(e[1])
            elif e[1] == seed:
                right_candidates.add(e[0])
        
        # Expand: find vertices that connect to all current left via uncovered
        changed = True
        while changed:
            changed = False
            new_right = set()
            for v in right_candidates:
                # Check if v connects to all in left_candidates via uncovered edges
                ok = True
                for u in left_candidates:
                    if (min(u,v), max(u,v)) not in uncovered:
                        ok = False
                        break
                if ok:
                    new_right.add(v)
            if len(new_right) < len(right_candidates):
                right_candidates = new_right
                changed = True
        
        if len(right_candidates) > 0 and len(left_candidates) * len(right_candidates) > best_count:
            # Verify it's a valid biclique
            valid = True
            for u in left_candidates:
                for v in right_candidates:
                    if (min(u,v), max(u,v)) not in uncovered:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_count = len(left_candidates) * len(right_candidates)
                best_left = set(left_candidates)
                best_right = set(right_candidates)
        
        # Try seed on right
        right_candidates2 = {seed}
        left_candidates2 = set()
        for e in uncovered:
            if e[1] == seed:
                left_candidates2.add(e[0])
            elif e[0] == seed:
                left_candidates2.add(e[1])
        
        changed = True
        while changed:
            changed = False
            new_left = set()
            for v in left_candidates2:
                ok = True
                for u in right_candidates2:
                    if (min(u,v), max(u,v)) not in uncovered:
                        ok = False
                        break
                if ok:
                    new_left.add(v)
            if len(new_left) < len(left_candidates2):
                left_candidates2 = new_left
                changed = True
        
        if len(left_candidates2) > 0 and len(left_candidates2) * len(right_candidates2) > best_count:
            valid = True
            for u in left_candidates2:
                for v in right_candidates2:
                    if (min(u,v), max(u,v)) not in uncovered:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_count = len(left_candidates2) * len(right_candidates2)
                best_left = set(left_candidates2)
                best_right = set(right_candidates2)
    
    if best_count == 0:
        # Fallback: pick a single edge
        e = next(iter(uncovered))
        return [e[0]], [e[1]]
    
    return list(best_left), list(best_right)

def _greedy_partition(adj, deadline):
    """Greedy: repeatedly extract max biclique from uncovered edges."""
    all_edges = set((min(u,v), max(u,v)) for u,v in P['edges'])
    uncovered = set(all_edges)
    bicliques = []
    
    while uncovered and time.time() < deadline:
        # Find a biclique
        left, right = _find_max_biclique(uncovered, adj, deadline)
        if left is None:
            # Pick remaining edges individually
            for e in sorted(uncovered):
                bicliques.append([e[0], e[1]])
            break
        
        # Get edges covered
        covered = set()
        for u in left:
            for v in right:
                if (min(u,v), max(u,v)) in uncovered:
                    covered.add((min(u,v), max(u,v)))
        
        if not covered:
            # Shouldn't happen, but safety
            e = next(iter(uncovered))
            bicliques.append([e[0], e[1]])
            uncovered.discard(e)
            continue
        
        bicliques.append([list(left), list(right)])
        uncovered -= covered
    
    # Handle any remaining single edges
    for e in sorted(uncovered):
        bicliques.append([e[0], e[1]])
    
    return bicliques

def _try_merge(bicliques, adj, deadline):
    """Try to merge pairs of bicliques if their union is still valid."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        i = 0
        while i < len(bicliques) and time.time() < deadline:
            j = i + 1
            while j < len(bicliques) and time.time() < deadline:
                # Try merging biclique i and j
                li, ri = bicliques[i][0], bicliques[i][1]
                lj, rj = bicliques[j][0], bicliques[j][1]
                
                # Check if we can merge: left_i + left_j on left, right_i + right_j on right
                # But we need to ensure all cross pairs are edges
                # Option 1: combine lefts, keep rights separate won't work for single biclique
                # Actually merging means: new_left = left_i ∪ left_j, new_right = right_i ∪ right_j
                # But that requires ALL cross pairs to be edges, which is too strict.
                
                # Better: try to see if we can put some vertices from one into the other
                # For now, try: can we add all of left_j to left_i (with right_i)?
                # i.e., is (left_i ∪ left_j) x right_i a biclique?
                new_left = list(set(li) | set(lj))
                # Check new_left x ri
                valid = True
                for u in new_left:
                    for v in ri:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and len(new_left) > len(li):
                    # Also need to check that we're not covering edges that belong to other bicliques
                    # For simplicity, just do the merge if valid
                    bicliques[i] = [new_left, list(ri)]
                    # Remove edges from j that are now covered
                    # Actually this is complex. Let's just try simple merges.
                    del bicliques[j]
                    changed = True
                    break
                
                # Try adding right_j to right_i
                new_right = list(set(ri) | set(rj))
                valid = True
                for u in li:
                    for v in new_right:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and len(new_right) > len(ri):
                    bicliques[i] = [list(li), new_right]
                    del bicliques[j]
                    changed = True
                    break
                
                j += 1
            i += 1
    
    return bicliques

def _validate(bicliques, adj):
    """Validate the partition: check all constraints."""
    n = P['n']
    all_edges = set((min(u,v), max(u,v)) for u,v in P['edges'])
    covered = set()
    
    for bic in bicliques:
        left, right = bic
        if not left or not right:
            return False
        # Check disjoint sides
        if set(left) & set(right):
            return False
        # Check all cross pairs are edges
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u,v), max(u,v))
                if e in covered:
                    return False  # Edge covered twice
                covered.add(e)
    
    # Check all edges covered
    if covered != all_edges:
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    adj = _build_adj()
    
    # Greedy partition
    bicliques = _greedy_partition(adj, deadline)
    
    # Try to merge
    bicliques = _try_merge(bicliques, adj, deadline)
    
    # Validate
    if not _validate(bicliques, adj):
        # Fallback to seed solution
        bicliques = [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    
    # Ensure proper types
    result = []
    for bic in bicliques:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
