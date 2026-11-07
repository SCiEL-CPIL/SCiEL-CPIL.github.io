"""Rewritten program: greedy + local-search biclique partition."""
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

n = P['n']
edge_set = set()
adj = [set() for _ in range(n)]
for a, b in P['edges']:
    edge_set.add((a, b))
    edge_set.add((b, a))
    adj[a].add(b)
    adj[b].add(a)
m = len(P['edges'])
all_edges = list(P['edges'])


def is_valid_biclique(L, R):
    """Check that every cross pair between L and R is an edge."""
    for u in L:
        for v in R:
            if v not in adj[u]:
                return False
    return True


def greedy_partition():
    """Greedy set cover: repeatedly pick biclique covering most uncovered edges."""
    uncovered = set(edge_set)
    bicliques = []
    while uncovered:
        best_biclique = None
        best_count = 0
        # Try bicliques based on each vertex as the left side
        for u in range(n):
            # Candidates for R: neighbors of u that have uncovered edges
            # For a biclique [L, R], L can include vertices that are common neighbors
            # Simple approach: L = [u], R = some subset of neighbors
            # Better: find maximal bicliques containing u
            # For each neighbor v of u, we can try L={u}, R={v} (trivial)
            # Or expand: L = {u} ∪ common neighbors, R = common neighbors
            # Let's try: for each subset pattern
            # Simplest effective: L = {u}, R = all neighbors of u with uncovered edges
            # But also try expanding L
            
            # Approach 1: single vertex left
            R_candidates = [v for v in adj[u] if (u, v) in uncovered]
            if not R_candidates:
                continue
            # Check if {u} x R_candidates is valid (it is, since all are neighbors)
            count = len(R_candidates)
            if count > best_count:
                best_count = count
                best_biclique = ([u], list(R_candidates))
            
            # Approach 2: expand left side with common neighbors
            # L = {u} ∪ S where S ⊆ N(u) ∩ (common neighbors with all of R)
            # For R = N(u) ∩ uncovered, find vertices w that are neighbors of all in R
            # and also neighbors of u
            if R_candidates:
                # Find vertices w that are neighbors of u and all of R_candidates
                # This gives a larger biclique [L, R] where L = {u} ∪ {w : w~u and w~v for all v in R}
                # Actually for a biclique [L,R], we need L×R ⊆ E
                # So if L = {u, w1, w2, ...}, each wi must be adjacent to all of R
                common_with_R = set(R_candidates)
                for v in R_candidates:
                    common_with_R = common_with_R & adj[v]
                # Vertices in L (besides u) must be in common_with_R and adjacent to u
                extra_L = [w for w in common_with_R if w != u and w in adj[u] and (w, u) in edge_set]
                # Wait, w must be adjacent to u (for L-L edges are allowed but w-u must be edge? No!)
                # Actually within-side edges are ALLOWED but not required.
                # The constraint is only: every cross pair (l,r) for l in L, r in R must be an edge.
                # So w in L just needs to be adjacent to every vertex in R.
                # w does NOT need to be adjacent to u.
                extra_L = [w for w in common_with_R if w != u]
                if extra_L:
                    L_full = [u] + extra_L
                    # Verify validity
                    if is_valid_biclique(L_full, R_candidates):
                        count2 = len(L_full) * len(R_candidates)
                        # But we only count uncovered edges
                        cnt = 0
                        for l in L_full:
                            for r in R_candidates:
                                if (l, r) in uncovered:
                                    cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_biclique = (L_full, list(R_candidates))
        
        if best_biclique is None:
            # Should not happen if uncovered is non-empty
            break
        
        L, R = best_biclique
        bicliques.append([L, R])
        for l in L:
            for r in R:
                uncovered.discard((l, r))
    
    return bicliques


def local_search(bicliques, time_limit):
    """Try to improve by merging or re-partitioning."""
    start = time.time()
    current = [list(b) for b in bicliques]
    
    while time.time() - start < time_limit:
        improved = False
        # Try merging pairs of bicliques
        for i in range(len(current)):
            if time.time() - start > time_limit:
                return current
            for j in range(i + 1, len(current)):
                L1, R1 = current[i]
                L2, R2 = current[j]
                # Try merging: new L = L1 ∪ L2, new R = R1 ∪ R2
                # Check if valid
                new_L = list(set(L1) | set(L2))
                new_R = list(set(R1) | set(R2))
                if is_valid_biclique(new_L, new_R):
                    # Check that all cross pairs are edges (already checked)
                    # Remove i and j, add merged
                    remaining = [current[k] for k in range(len(current)) if k != i and k != j]
                    remaining.append([new_L, new_R])
                    current = remaining
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try splitting a biclique to free up edges for better packing
            # This is complex; try a different heuristic
            # Try: for each biclique, see if we can move some edges to other bicliques
            # and reduce total count
            best_move = None
            for i in range(len(current)):
                L, R = current[i]
                # Try removing one vertex from L and see if remaining is still valid
                # and if the removed vertex's edges can be covered by other bicliques
                for l in L:
                    new_L = [x for x in L if x != l]
                    if not new_L:
                        continue
                    # The edges from l to R are now uncovered
                    # Check if they can be covered by existing other bicliques
                    edges_to_cover = set()
                    for r in R:
                        edges_to_cover.add((l, r))
                    covered = True
                    for k in range(len(current)):
                        if k == i:
                            continue
                        Lk, Rk = current[k]
                        for e in list(edges_to_cover):
                            if e[0] in Lk and e[1] in Rk:
                                edges_to_cover.discard(e)
                            elif e[1] in Lk and e[0] in Rk:
                                edges_to_cover.discard(e)
                    if edges_to_cover:
                        covered = False
                    if covered and len(new_L) > 0:
                        # This move doesn't reduce count, skip for now
                        pass
                # Try removing a vertex from R similarly
                for r in R:
                    new_R = [x for x in R if x != r]
                    if not new_R:
                        continue
                    edges_to_cover = set()
                    for l in L:
                        edges_to_cover.add((l, r))
                    covered = True
                    for k in range(len(current)):
                        if k == i:
                            continue
                        Lk, Rk = current[k]
                        for e in list(edges_to_cover):
                            if e[0] in Lk and e[1] in Rk:
                                edges_to_cover.discard(e)
                            elif e[1] in Lk and e[0] in Rk:
                                edges_to_cover.discard(e)
                    if edges_to_cover:
                        covered = False
                    if covered and len(new_R) > 0:
                        pass
            
            break
    
    return current


def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin before 90s kill
    
    # Phase 1: Greedy
    bicliques = greedy_partition()
    
    if time.time() > deadline - 5:
        # Not enough time for local search
        return [[list(L), list(R)] for L, R in bicliques]
    
    # Phase 2: Local search
    bicliques = local_search(bicliques, deadline - time.time() - 2.0)
    
    # Ensure proper types
    result = []
    for L, R in bicliques:
        result.append([list(int(x) for x in L), list(int(x) for x in R)])
    
    return result
