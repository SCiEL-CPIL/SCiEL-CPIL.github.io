"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_biclique(adj, remaining_edges, time_limit=0.5):
    """Find a large biclique in the subgraph defined by remaining_edges.
    Uses greedy with multiple anchors and iterative refinement.
    """
    if not remaining_edges:
        return None, None
    
    n = P['n']
    start_time = time.time()
    
    rem_set = set()
    for u, v in remaining_edges:
        rem_set.add((u, v))
        rem_set.add((v, u))
    
    deg = [0] * n
    for u, v in remaining_edges:
        deg[u] += 1
        deg[v] += 1
    
    candidates = [v for v in range(n) if deg[v] > 0]
    candidates.sort(key=lambda x: -deg[x])
    
    best_left = None
    best_right = None
    best_edge_count = 0
    
    max_anchors = min(len(candidates), 40)
    
    for anchor in candidates[:max_anchors]:
        if time.time() - start_time > time_limit:
            break
        
        neighbors = set()
        for v in adj[anchor]:
            if (anchor, v) in rem_set:
                neighbors.add(v)
        
        if not neighbors:
            continue
        
        left = {anchor}
        right = set(neighbors)
        
        # Iterative refinement
        changed = True
        while changed:
            changed = False
            
            if left:
                new_right = set(adj[list(left)[0]])
                for u in left:
                    if u != list(left)[0]:
                        new_right &= adj[u]
                new_right &= right
                if new_right != right:
                    right = new_right
                    changed = True
            
            if right:
                new_left = set(adj[list(right)[0]])
                for v in right:
                    if v != list(right)[0]:
                        new_left &= adj[v]
                new_left &= left
                if new_left != left:
                    left = new_left
                    changed = True
        
        if not left or not right:
            continue
        
        # Expand
        for u in candidates:
            if u in left:
                continue
            if all(u in adj[v] for v in right):
                if any((u, v) in rem_set for v in right):
                    left.add(u)
        
        for v in candidates:
            if v in right:
                continue
            if all(v in adj[u] for u in left):
                if any((u, v) in rem_set for u in left):
                    right.add(v)
        
        if not left or not right:
            continue
        
        edge_count = 0
        for u in left:
            for v in right:
                if (u, v) in rem_set:
                    edge_count += 1
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_left = sorted(left)
            best_right = sorted(right)
    
    return best_left, best_right

def _validate_bicliques(bicliques, edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                edge = tuple(sorted((u, v)))
                if edge not in edges_set:
                    return False
                if edge in covered:
                    return False
                covered.add(edge)
    return covered == edges_set

def _try_merge_pair(bic_i, bic_j, edges_set):
    """Try to merge two bicliques into one. Returns (merged, new_bic_j) or (None, None)."""
    li, ri = bic_i[0], bic_i[1]
    lj, rj = bic_j[0], bic_j[1]
    
    # Try all 4 combinations of left/right sides
    combos = [
        (set(li) | set(lj), set(ri) | set(rj)),
        (set(li) | set(rj), set(ri) | set(lj)),
        (set(ri) | set(lj), set(li) | set(rj)),
        (set(ri) | set(rj), set(li) | set(lj)),
    ]
    
    # Compute current covered edges for both bicliques
    covered_i = set()
    for u in li:
        for v in ri:
            covered_i.add(tuple(sorted((u, v))))
    covered_j = set()
    for u in lj:
        for v in rj:
            covered_j.add(tuple(sorted((u, v))))
    target = covered_i | covered_j
    
    for new_left, new_right in combos:
        if new_left & new_right:
            continue
        if not new_left or not new_right:
            continue
        
        # Check all cross pairs are edges
        valid = True
        for u in new_left:
            for v in new_right:
                if tuple(sorted((u, v))) not in edges_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Check coverage matches
        covered_new = set()
        for u in new_left:
            for v in new_right:
                covered_new.add(tuple(sorted((u, v))))
        
        if covered_new == target:
            return [sorted(new_left), sorted(new_right)]
    
    return None

def _try_absorb_vertex(bic_target, bic_source, edges_set):
    """Try to move a vertex from bic_source to bic_target.
    Returns (new_target, new_source) or (None, None)."""
    tl, tr = bic_target[0], bic_target[1]
    sl, sr = bic_source[0], bic_source[1]
    
    # Try moving a vertex from sl to tl
    for u in sl:
        if u in tl:
            continue
        # Check u connects to all of tr
        if all(tuple(sorted((u, v))) in edges_set for v in tr):
            new_tl = tl + [u]
            new_sl = [x for x in sl if x != u]
            if not new_sl:
                continue
            # Check new edges don't conflict (all must be in edges_set)
            # And that removing u from sl doesn't break source biclique
            # Source biclique is just (new_sl, sr) - need to verify it's still valid
            # Actually we need the FULL partition to be valid, so let's just return the candidate
            # and let the caller validate
            return [new_tl, tr], [new_sl, sr]
    
    # Try moving a vertex from sr to tr
    for v in sr:
        if v in tr:
            continue
        if all(tuple(sorted((u, v))) in edges_set for u in tl):
            new_tr = tr + [v]
            new_sr = [x for x in sr if x != v]
            if not new_sr:
                continue
            return [tl, new_tr], [sl, new_sr]
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    adj = _build_adjacency()
    edges_set = set(tuple(sorted(e)) for e in P['edges'])
    
    def star_partition():
        result = []
        for u in range(P['n']):
            neighbors = sorted(adj[u])
            if neighbors:
                result.append([neighbors, [u]])
        return result
    
    fallback = star_partition()
    best = fallback
    best_count = len(fallback)
    
    # Greedy biclique partition
    remaining = set(edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        left, right = _find_biclique(adj, remaining, time_limit=1.0)
        
        if left is None or right is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        rem_set_local = set()
        for u, v in remaining:
            rem_set_local.add((u, v))
            rem_set_local.add((v, u))
        
        valid_left = []
        for u in left:
            if all((u, v) in rem_set_local for v in right):
                valid_left.append(u)
        
        valid_right = []
        for v in right:
            if all((u, v) in rem_set_local for u in valid_left):
                valid_right.append(v)
        
        if not valid_left or not valid_right:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        for u in valid_left:
            for v in valid_right:
                edge = tuple(sorted((u, v)))
                remaining.discard(edge)
        
        bicliques.append([valid_left, valid_right])
    
    for edge in remaining:
        u, v = edge
        bicliques.append([[u], [v]])
    
    if _validate_bicliques(bicliques, edges_set):
        if len(bicliques) < best_count:
            best = bicliques
            best_count = len(bicliques)
    else:
        best = fallback
    
    # Merge pass: try to merge pairs of bicliques
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            n_b = len(best)
            for i in range(n_b):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, n_b):
                    if time.time() > deadline - 10:
                        break
                    merged = _try_merge_pair(best[i], best[j], edges_set)
                    if merged is not None:
                        best[i] = merged
                        del best[j]
                        improved = True
                        break
                if improved:
                    break
        
        if _validate_bicliques(best, edges_set):
            best_count = len(best)
        else:
            best = star_partition()
            best_count = len(fallback)
    
    # Absorb pass: try to move vertices between bicliques to reduce count
    if time.time() < deadline - 5:
        polish_improved = True
        while polish_improved and time.time() < deadline - 5:
            polish_improved = False
            n_b = len(best)
            for i in range(n_b):
                if time.time() > deadline - 5:
                    break
                for j in range(n_b):
                    if i == j or time.time() > deadline - 5:
                        continue
                    new_i, new_j = _try_absorb_vertex(best[i], best[j], edges_set)
                    if new_i is not None and new_j is not None:
                        # Check if this reduces count (source becomes empty)
                        # Or if it enables a future merge
                        test_best = [list(x) for x in best]
                        test_best[i] = new_i
                        test_best[j] = new_j
                        # Remove empty bicliques
                        test_best = [b for b in test_best if b[0] and b[1]]
                        if _validate_bicliques(test_best, edges_set):
                            if len(test_best) < len(best):
                                best = test_best
                                polish_improved = True
                                break
                    # Also try the reverse direction
                    new_j2, new_i2 = _try_absorb_vertex(best[j], best[i], edges_set)
                    if new_j2 is not None and new_i2 is not None:
                        test_best = [list(x) for x in best]
                        test_best[j] = new_j2
                        test_best[i] = new_i2
                        test_best = [b for b in test_best if b[0] and b[1]]
                        if _validate_bicliques(test_best, edges_set):
                            if len(test_best) < len(best):
                                best = test_best
                                polish_improved = True
                                break
                if polish_improved:
                    break
    
    # Final merge pass after absorbs
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n_b = len(best)
            for i in range(n_b):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, n_b):
                    if time.time() > deadline - 2:
                        break
                    merged = _try_merge_pair(best[i], best[j], edges_set)
                    if merged is not None:
                        best[i] = merged
                        del best[j]
                        improved = True
                        break
                if improved:
                    break
        
        if _validate_bicliques(best, edges_set):
            best_count = len(best)
        else:
            best = star_partition()
    
    # Ensure types
    result = []
    for bic in best:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result

# EVOLVE-BLOCK-END
