"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_biclique_in_subgraph(adj, remaining_edge_set, n, time_limit=0.3):
    """Find a large biclique within the subgraph defined by remaining_edge_set.
    Uses multi-anchor iterative refinement + expansion.
    Returns (left_set, right_set) or (None, None).
    """
    if not remaining_edge_set:
        return None, None
    
    start_t = time.time()
    
    # Build degree from remaining edges
    deg = [0] * n
    rem_adj = [set() for _ in range(n)]
    for (u, v) in remaining_edge_set:
        deg[u] += 1
        deg[v] += 1
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    candidates = [v for v in range(n) if deg[v] > 0]
    candidates.sort(key=lambda x: -deg[x])
    
    best_left = None
    best_right = None
    best_count = 0
    
    max_anchors = min(len(candidates), 30)
    
    for anchor in candidates[:max_anchors]:
        if time.time() - start_t > time_limit:
            break
        
        neighbors = rem_adj[anchor]
        if not neighbors:
            continue
        
        # Start: left={anchor}, right=neighbors
        left = {anchor}
        right = set(neighbors)
        
        # Iterative refinement (Purify)
        changed = True
        while changed:
            changed = False
            new_right = set()
            for v in right:
                if all(v in adj[u] for u in left):
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            new_left = set()
            for u in left:
                if all(u in adj[v] for v in right):
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        if not left or not right:
            continue
        
        # Expansion: add vertices that are adjacent to all on the other side
        # and have at least one remaining edge to the other side
        expanded = True
        while expanded:
            expanded = False
            for u in candidates:
                if u not in left and u in rem_adj[anchor] or u in rem_adj[anchor]:
                    pass
                if u not in left:
                    all_adj = all(u in adj[v] for v in right)
                    if all_adj:
                        # Check if u has any remaining edge to some v in right
                        has_edge = any(_edge_key(u, v) in remaining_edge_set for v in right)
                        if has_edge:
                            left.add(u)
                            expanded = True
        
        expanded = True
        while expanded:
            expanded = False
            for v in candidates:
                if v not in right:
                    all_adj = all(v in adj[u] for u in left)
                    if all_adj:
                        has_edge = any(_edge_key(u, v) in remaining_edge_set for u in left)
                        if has_edge:
                            right.add(v)
                            expanded = True
        
        if not left or not right:
            continue
        
        # Count edges covered
        count = 0
        for u in left:
            for v in right:
                if _edge_key(u, v) in remaining_edge_set:
                    count += 1
        
        if count > best_count:
            best_count = count
            best_left = set(left)
            best_right = set(right)
    
    if best_left is None or best_right is None:
        return None, None
    
    return best_left, best_right

def _can_merge(b1, b2, adj):
    """Check if bicliques can be merged."""
    L1, R1 = set(b1[0]), set(b1[1])
    L2, R2 = set(b2[0]), set(b2[1])
    for u in L2:
        for v in R1:
            if v not in adj[u]:
                return False
    for u in L1:
        for v in R2:
            if v not in adj[u]:
                return False
    return True

def _merge_bicliques(b1, b2):
    L1, R1 = list(b1[0]), list(b1[1])
    L2, R2 = list(b2[0]), list(b2[1])
    newL = list(set(L1) | set(L2))
    newR = list(set(R1) | set(R2))
    return [newL, newR]

def _greedy_partition(adj, edge_set, n, deadline):
    """Greedy max-biclique extraction: repeatedly find largest biclique in remaining edges."""
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        left, right = _find_biclique_in_subgraph(adj, remaining, n, time_limit=0.3)
        
        if left is None or right is None or not left or not right:
            # Fallback: take one edge
            e = next(iter(remaining))
            u, v = e
            bicliques.append([[u], [v]])
            remaining.discard(e)
            continue
        
        # Remove covered edges
        for u in left:
            for v in right:
                remaining.discard(_edge_key(u, v))
        
        bicliques.append([list(left), list(right)])
    
    return bicliques

def _merge_pass(bicliques, adj, deadline):
    """Repeatedly merge pairs of bicliques where possible."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        new_bicliques = []
        used = [False] * len(bicliques)
        for i in range(len(bicliques)):
            if used[i]:
                continue
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if used[j]:
                    continue
                if _can_merge(bicliques[i], bicliques[j], adj):
                    new_bicliques.append(_merge_bicliques(bicliques[i], bicliques[j]))
                    used[i] = True
                    used[j] = True
                    improved = True
                    break
            if not used[i]:
                new_bicliques.append(bicliques[i])
        bicliques = new_bicliques
    return bicliques

def _validate(bicliques, edge_set, adj, n):
    """Check validity: all cross pairs are edges, partition exactly, nonempty disjoint sides."""
    covered = set()
    for b in bicliques:
        L, R = b[0], b[1]
        if not L or not R:
            return False
        L_set = set(L)
        R_set = set(R)
        if L_set & R_set:
            return False
        for u in L:
            for v in R:
                if u == v:
                    return False
                e = _edge_key(u, v)
                if e not in adj[u]:  # adj check
                    return False
                if e not in edge_set:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == edge_set

def _local_search(bicliques, adj, edge_set, n, deadline):
    """Try to improve by: (1) splitting a biclique to help merge others, (2) moving vertices between bicliques."""
    best = bicliques
    best_count = len(bicliques)
    
    # Try: for each biclique, try removing a vertex from one side and see if it helps
    # This is complex; instead try a simpler approach:
    # Try to find if any biclique can be split into two, and those two can merge with others
    
    # Alternative: try re-partitioning small subsets
    # For now, just do the merge pass again with different ordering
    
    # Try: for each pair, if they can't merge, try removing one vertex from one to make them mergeable
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                if not _can_merge(bicliques[i], bicliques[j], adj):
                    # Try removing a vertex from biclique i's left or right
                    Li, Ri = list(bicliques[i][0]), list(bicliques[i][1])
                    Lj, Rj = list(bicliques[j][0]), list(bicliques[j][1])
                    
                    # Try removing each vertex from Li
                    for k, u in enumerate(Li):
                        Li_new = Li[:k] + Li[k+1:]
                        if not Li_new:
                            continue
                        b1_new = [Li_new, Ri]
                        if _can_merge(b1_new, bicliques[j], adj):
                            # Check: does removing u from Li leave valid biclique?
                            # u must not be in any other biclique's cross pairs with Ri
                            merged = _merge_bicliques(b1_new, bicliques[j])
                            # Verify merged is valid
                            Lm, Rm = set(merged[0]), set(merged[1])
                            valid = True
                            for a in Lm:
                                for b in Rm:
                                    if b not in adj[a]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                # Also check u's edges are still covered (u was in Li, so u-Ri edges are in b1)
                                # After removal, u is no longer in any biclique -> its edges uncovered
                                # So this only works if u had no edges to Ri... but it did.
                                # Actually this approach is flawed. Let's skip.
                                pass
                    
                    # Try removing from Ri
                    for k, v in enumerate(Ri):
                        Ri_new = Ri[:k] + Ri[k+1:]
                        if not Ri_new:
                            continue
                        b1_new = [Li, Ri_new]
                        if _can_merge(b1_new, bicliques[j], adj):
                            Lm, Rm = set(Li) | set(Lj), set(Ri_new) | set(Rj)
                            valid = True
                            for a in Lm:
                                for b in Rm:
                                    if b not in adj[a]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                # v's edges to Li are now uncovered -> invalid
                                pass
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin (90s limit)
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    edge_set = set(_edge_key(u, v) for u, v in edges)
    
    # Strategy 1: Greedy max-biclique extraction
    bicliques1 = _greedy_partition(adj, edge_set, n, deadline)
    
    # Apply merge pass
    bicliques1 = _merge_pass(bicliques1, adj, deadline)
    
    # Validate
    if not _validate(bicliques1, edge_set, adj, n):
        # Fallback: star partition
        bicliques1 = []
        for u in range(n):
            neighbors = sorted(adj[u])
            if neighbors:
                bicliques1.append([neighbors, [u]])
    
    # Strategy 2: Try different anchor orderings for greedy extraction
    if time.time() < deadline - 10:
        bicliques2 = _greedy_partition(adj, edge_set, n, deadline)
        bicliques2 = _merge_pass(bicliques2, adj, deadline)
        if _validate(bicliques2, edge_set, adj, n) and len(bicliques2) < len(bicliques1):
            bicliques1 = bicliques2
    
    # Final: ensure proper format
    result = []
    for b in bicliques1:
        L = [int(x) for x in b[0]]
        R = [int(x) for x in b[1]]
        result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
