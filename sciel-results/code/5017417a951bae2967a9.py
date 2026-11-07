"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _common_neighbors(adj, left, right):
    """Find vertices that are adjacent to all in both left and right."""
    n = P['n']
    candidates = []
    for v in range(n):
        if v in left or v in right:
            continue
        ok = True
        for u in left:
            if v not in adj[u]:
                ok = False
                break
        if not ok:
            continue
        for u in right:
            if v not in adj[u]:
                ok = False
                break
        if ok:
            candidates.append(v)
    return candidates

def _find_biclique(adj, remaining_edges, time_limit=2.0):
    """Find a large biclique using multiple anchor strategies and iterative refinement."""
    if not remaining_edges:
        return None, None
    
    n = P['n']
    start_time = time.time()
    
    # Build remaining-neighbor sets for efficiency
    rem_adj = [set() for _ in range(n)]
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    deg = [len(rem_adj[v]) for v in range(n)]
    candidates = [v for v in range(n) if deg[v] > 0]
    candidates.sort(key=lambda x: -deg[x])
    
    best_left = None
    best_right = None
    best_edge_count = 0
    
    max_anchors = min(len(candidates), 40)
    
    for anchor in candidates[:max_anchors]:
        if time.time() - start_time > time_limit:
            break
        
        # Strategy A: anchor on left, neighbors on right
        right = set(rem_adj[anchor])
        left = {anchor}
        
        # Iteratively refine
        changed = True
        while changed:
            changed = False
            # Filter right: must be adjacent to all in left (via original graph)
            new_right = set()
            for v in right:
                all_adj = True
                for u in left:
                    if v not in adj[u]:
                        all_adj = False
                        break
                if all_adj:
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            # Filter left: must be adjacent to all in right
            new_left = set()
            for u in left:
                all_adj = True
                for v in right:
                    if u not in adj[v]:
                        all_adj = False
                        break
                if all_adj:
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        if not left or not right:
            continue
        
        # Expand: add vertices adjacent to all in opposite side AND having remaining edges
        expanded = True
        while expanded:
            expanded = False
            for u in candidates:
                if u not in left and u not in right and deg[u] > 0:
                    all_adj = True
                    for v in right:
                        if u not in adj[v]:
                            all_adj = False
                            break
                    if all_adj and rem_adj[u] & right:
                        left.add(u)
                        expanded = True
        
        expanded = True
        while expanded:
            expanded = False
            for v in candidates:
                if v not in left and v not in right and deg[v] > 0:
                    all_adj = True
                    for u in left:
                        if v not in adj[u]:
                            all_adj = False
                            break
                    if all_adj and rem_adj[v] & left:
                        right.add(v)
                        expanded = True
        
        if not left or not right:
            continue
        
        # Count edges covered
        edge_count = 0
        for u in left:
            edge_count += len(rem_adj[u] & right)
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_left = sorted(left)
            best_right = sorted(right)
        
        # Strategy B: anchor on right, neighbors on left
        left2 = set(rem_adj[anchor])
        right2 = {anchor}
        
        changed = True
        while changed:
            changed = False
            new_left2 = set()
            for v in left2:
                all_adj = True
                for u in right2:
                    if v not in adj[u]:
                        all_adj = False
                        break
                if all_adj:
                    new_left2.add(v)
            if new_left2 != left2:
                left2 = new_left2
                changed = True
            
            new_right2 = set()
            for u in right2:
                all_adj = True
                for v in left2:
                    if u not in adj[v]:
                        all_adj = False
                        break
                if all_adj:
                    new_right2.add(u)
            if new_right2 != right2:
                right2 = new_right2
                changed = True
        
        if not left2 or not right2:
            continue
        
        expanded = True
        while expanded:
            expanded = False
            for u in candidates:
                if u not in left2 and u not in right2 and deg[u] > 0:
                    all_adj = True
                    for v in right2:
                        if u not in adj[v]:
                            all_adj = False
                            break
                    if all_adj and rem_adj[u] & right2:
                        left2.add(u)
                        expanded = True
        
        expanded = True
        while expanded:
            expanded = False
            for v in candidates:
                if v not in left2 and v not in right2 and deg[v] > 0:
                    all_adj = True
                    for u in left2:
                        if v not in adj[u]:
                            all_adj = False
                            break
                    if all_adj and rem_adj[v] & left2:
                        right2.add(v)
                        expanded = True
        
        if not left2 or not right2:
            continue
        
        edge_count2 = 0
        for u in left2:
            edge_count2 += len(rem_adj[u] & right2)
        
        if edge_count2 > best_edge_count:
            best_edge_count = edge_count2
            best_left = sorted(left2)
            best_right = sorted(right2)
    
    return best_left, best_right

def _validate_bicliques(bicliques, edges_set):
    """Validate that bicliques form a valid partition."""
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

def _merge_pass(bicliques, adj, edges_set, deadline):
    """Try to merge pairs of bicliques using multiple orientation strategies."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try all 4 merge orientations
                merged = None
                for (new_l, new_r) in [
                    (set(li) | set(lj), set(ri) | set(rj)),
                    (set(li) | set(lj), set(ri) & set(rj)),
                    (set(li) & set(lj), set(ri) | set(rj)),
                    (set(li) | set(rj), set(ri) & set(lj)),
                    (set(lj) | set(ri), set(rj) & set(li)),
                    (set(li) | set(rj), set(rj) & set(li)),
                    (set(lj) | set(ri), set(ri) & set(rj)),
                ]:
                    if not new_l or not new_r:
                        continue
                    if new_l & new_r:
                        continue
                    # Verify all cross pairs are edges in original graph
                    valid = True
                    for u in new_l:
                        for v in new_r:
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    # Check that merged covers exactly the union of edges from i and j
                    covered_i = set()
                    for u in li:
                        for v in ri:
                            covered_i.add((min(u, v), max(u, v)))
                    covered_j = set()
                    for u in lj:
                        for v in rj:
                            covered_j.add((min(u, v), max(u, v)))
                    covered_new = set()
                    for u in new_l:
                        for v in new_r:
                            covered_new.add((min(u, v), max(u, v)))
                    if covered_new == covered_i | covered_j:
                        merged = (sorted(new_l), sorted(new_r))
                        break
                if merged:
                    bicliques[i] = merged
                    del bicliques[j]
                    changed = True
                    break
            if changed:
                break
    return bicliques

def _local_search(bicliques, adj, edges_set, deadline):
    """Try to reassign edges between bicliques to reduce count."""
    # Try removing vertices from bicliques and reassigning
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            li, ri = bicliques[i]
            # Try removing a vertex from left and seeing if we can merge
            for u in li[:]:
                if time.time() > deadline:
                    break
                new_li = [x for x in li if x != u]
                if not new_li:
                    continue
                # Check if the edges from u to ri can be absorbed by other bicliques
                u_edges = set()
                for v in ri:
                    u_edges.add((min(u, v), max(u, v)))
                # Check if all u_edges are covered by other bicliques
                all_covered = True
                for e in u_edges:
                    found = False
                    for k, (lk, rk) in enumerate(bicliques):
                        if k == i:
                            continue
                        if e[0] in lk and e[1] in rk:
                            found = True
                            break
                        if e[1] in lk and e[0] in rk:
                            found = True
                            break
                    if not found:
                        all_covered = False
                        break
                if all_covered:
                    bicliques[i] = (new_li, ri)
                    improved = True
                    break
            if improved:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    adj = _build_adj()
    edges_set = set(tuple(sorted(e)) for e in P['edges'])
    
    # Fallback: star decomposition
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
    
    # Greedy biclique partition with multiple anchor strategies
    remaining = set(edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline - 20:
            break
        
        left, right = _find_biclique(adj, remaining, time_limit=1.5)
        
        if left is None or right is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        # Verify and shrink to only remaining edges
        valid_left = []
        for u in left:
            if all((min(u, v), max(u, v)) in remaining for v in right):
                valid_left.append(u)
        
        valid_right = []
        for v in right:
            if all((min(u, v), max(u, v)) in remaining for u in valid_left):
                valid_right.append(v)
        
        if not valid_left or not valid_right:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        # Remove covered edges
        for u in valid_left:
            for v in valid_right:
                edge = (min(u, v), max(u, v))
                remaining.discard(edge)
        
        bicliques.append([valid_left, valid_right])
    
    # Handle leftover edges
    for edge in remaining:
        u, v = edge
        bicliques.append([[u], [v]])
    
    # Validate
    if _validate_bicliques(bicliques, edges_set):
        if len(bicliques) < best_count:
            best = bicliques
            best_count = len(bicliques)
    else:
        best = fallback
    
    # Merge pass
    if time.time() < deadline - 10:
        best = _merge_pass(best, adj, edges_set, deadline - 5)
        if _validate_bicliques(best, edges_set):
            best_count = len(best)
        else:
            best = fallback
    
    # Local search: try to shrink bicliques
    if time.time() < deadline - 5:
        best = _local_search(best, adj, edges_set, deadline - 2)
        if _validate_bicliques(best, edges_set):
            best_count = len(best)
        else:
            best = fallback
    
    # Ensure types are correct
    result = []
    for bic in best:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result

# EVOLVE-BLOCK-END
