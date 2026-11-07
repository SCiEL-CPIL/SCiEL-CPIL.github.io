"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_maximal_bicliques(adj, edges_set, n):
    """Find maximal bicliques using a greedy expansion approach with deduplication."""
    maximal_bicliques = []
    seen_keys = set()
    
    edge_list = list(edges_set)
    
    for u, v in edge_list:
        L = {u}
        R = {v}
        
        # Greedy expansion to a fixed point
        changed = True
        while changed:
            changed = False
            
            # Try adding to L
            for x in range(n):
                if x in L or x in R:
                    continue
                if all(x in adj[r] for r in R):
                    L.add(x)
                    changed = True
            
            # Try adding to R
            for x in range(n):
                if x in L or x in R:
                    continue
                if all(x in adj[l] for l in L):
                    R.add(x)
                    changed = True
        
        if not L or not R:
            continue
        
        # Verify all cross pairs are edges
        valid = True
        for a in L:
            for b in R:
                if b not in adj[a]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Deduplicate
        key = (frozenset(L), frozenset(R))
        key2 = (frozenset(R), frozenset(L))
        if key not in seen_keys and key2 not in seen_keys:
            seen_keys.add(key)
            seen_keys.add(key2)
            maximal_bicliques.append((sorted(L), sorted(R)))
    
    return maximal_bicliques

def _solve_exact_cover(bicliques, edges_set, time_limit=5.0):
    """Solve the exact cover problem using recursive backtracking.
    
    Each biclique covers a set of edges. We need to select a subset of bicliques
    such that every edge is covered exactly once.
    """
    start_time = time.time()
    
    # Map each biclique to the set of edges it covers
    biclique_edges = []
    for L, R in bicliques:
        edges = set()
        for u in L:
            for v in R:
                edges.add(tuple(sorted((u, v))))
        biclique_edges.append(edges)
    
    # Filter out bicliques that don't cover any edges
    valid_indices = [i for i, e in enumerate(biclique_edges) if e]
    biclique_edges = [biclique_edges[i] for i in valid_indices]
    bicliques = [bicliques[i] for i in valid_indices]
    
    # Remove dominated bicliques: if biclique A's edges are a subset of biclique B's edges,
    # and A has fewer or equal edges, prefer B (larger coverage)
    # Actually, for exact cover, we want to keep all, but we can prune some
    
    # Precompute which bicliques cover each edge
    edge_to_bicliques = {}
    for i, edges in enumerate(biclique_edges):
        for e in edges:
            if e not in edge_to_bicliques:
                edge_to_bicliques[e] = []
            edge_to_bicliques[e].append(i)
    
    all_edges = set(edges_set)
    
    best_solution = None
    best_count = float('inf')
    
    def backtrack(covered, remaining, current, depth=0):
        nonlocal best_solution, best_count
        
        if time.time() - start_time > time_limit:
            return
        
        # Pruning: if current count already >= best, stop
        if len(current) >= best_count:
            return
        
        if not remaining:
            if len(current) < best_count:
                best_count = len(current)
                best_solution = list(current)
            return
        
        # Find the edge with fewest candidate bicliques (MRV heuristic)
        min_edge = None
        min_candidates = float('inf')
        for e in remaining:
            cands = [b for b in edge_to_bicliques.get(e, []) if b not in covered]
            if len(cands) < min_candidates:
                min_candidates = len(cands)
                min_edge = e
                if min_candidates == 0:
                    return  # Dead end
        
        if min_candidates == 0:
            return
        
        # Try each biclique that covers min_edge
        candidates = [b for b in edge_to_bicliques.get(min_edge, []) if b not in covered]
        
        # Sort by number of edges covered (prefer larger bicliques first)
        candidates.sort(key=lambda b: -len(biclique_edges[b]))
        
        for b in candidates:
            if b in covered:
                continue
            
            # Check if adding b conflicts with already covered edges
            conflict = False
            for e in biclique_edges[b]:
                if e in covered:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Add b and recurse
            new_covered = covered | set(biclique_edges[b])
            new_remaining = remaining - set(biclique_edges[b])
            current.append(b)
            
            backtrack(new_covered, new_remaining, current, depth + 1)
            
            current.pop()
    
    backtrack(set(), all_edges, [], 0)
    
    if best_solution is not None:
        return [bicliques[i] for i in best_solution]
    
    return None

def _find_biclique(adj, remaining_edges, time_limit=0.5):
    """Find a large biclique in the subgraph defined by remaining_edges.
    
    Uses a greedy approach with multiple starting points and iterative refinement.
    """
    if not remaining_edges:
        return None, None
    
    n = P['n']
    start_time = time.time()
    
    # Build a set for O(1) edge lookup
    rem_set = set()
    for u, v in remaining_edges:
        rem_set.add((u, v))
        rem_set.add((v, u))
    
    # Build degree counts from remaining edges
    deg = [0] * n
    for u, v in remaining_edges:
        deg[u] += 1
        deg[v] += 1
    
    # Get vertices with positive degree, sorted by degree descending
    candidates = [v for v in range(n) if deg[v] > 0]
    candidates.sort(key=lambda x: -deg[x])
    
    best_left = None
    best_right = None
    best_edge_count = 0
    
    # Try multiple anchors
    max_anchors = min(len(candidates), 40)
    
    for anchor in candidates[:max_anchors]:
        if time.time() - start_time > time_limit:
            break
        
        # Neighbors of anchor in remaining edges (O(deg) via adjacency)
        neighbors = set()
        for v in adj[anchor]:
            if (anchor, v) in rem_set:
                neighbors.add(v)
        
        if not neighbors:
            continue
        
        # Start with anchor on left, neighbors on right
        left = {anchor}
        right = set(neighbors)
        
        # Iteratively refine until stable using common-neighbor intersection
        changed = True
        while changed:
            changed = False
            
            # Filter right: must be adjacent to all in left (intersection of adj[u] for u in left)
            if left:
                new_right = set(adj[list(left)[0]])
                for u in left:
                    if u != list(left)[0]:
                        new_right &= adj[u]
                new_right &= right
                if new_right != right:
                    right = new_right
                    changed = True
            
            # Filter left: must be adjacent to all in right
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
        
        # Try to expand: add vertices that are adjacent to all in opposite side AND have remaining edges to that side
        # Expand left
        for u in candidates:
            if u in left:
                continue
            # Check if u is adjacent to all in right
            if all(u in adj[v] for v in right):
                # Check if u has any remaining edges to right
                if any((u, v) in rem_set for v in right):
                    left.add(u)
        
        # Expand right
        for v in candidates:
            if v in right:
                continue
            # Check if v is adjacent to all in left
            if all(v in adj[u] for u in left):
                # Check if v has any remaining edges to left
                if any((u, v) in rem_set for u in left):
                    right.add(v)
        
        if not left or not right:
            continue
        
        # Count edges this biclique covers from remaining_edges
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

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    adj = _build_adjacency()
    edges_set = set(tuple(sorted(e)) for e in P['edges'])
    n = P['n']
    
    # Fallback: star decomposition (one biclique per vertex)
    def star_partition():
        result = []
        for u in range(n):
            neighbors = sorted(adj[u])
            if neighbors:
                result.append([neighbors, [u]])
        return result
    
    fallback = star_partition()
    best = fallback
    best_count = len(fallback)
    
    # First, try the exact cover approach using maximal bicliques
    if time.time() < deadline - 20:
        maximal_bicliques = _find_maximal_bicliques(adj, edges_set, n)
        
        if maximal_bicliques:
            # Solve exact cover
            solution = _solve_exact_cover(maximal_bicliques, edges_set, time_limit=15.0)
            
            if solution is not None:
                # Validate the solution
                if _validate_bicliques(solution, edges_set):
                    if len(solution) < best_count:
                        best = solution
                        best_count = len(solution)
    
    # If exact cover didn't work or wasn't fast enough, use greedy as backup
    if len(best) == best_count and best_count == len(fallback):
        # Greedy biclique partition
        remaining = set(edges_set)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            left, right = _find_biclique(adj, remaining, time_limit=1.0)
            
            if left is None or right is None:
                # Fallback: take one edge
                u, v = next(iter(remaining))
                bicliques.append([[u], [v]])
                remaining.discard((u, v))
                continue
            
            # Verify and shrink to only remaining edges
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
            
            # Remove covered edges
            for u in valid_left:
                for v in valid_right:
                    edge = tuple(sorted((u, v)))
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
            # Use fallback
            best = fallback
    
    # Try to improve by merging small bicliques and absorbing edges
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            n_bicliques = len(best)
            for i in range(n_bicliques):
                if time.time() > deadline - 10:
                    break
                li, ri = best[i][0], best[i][1]
                for j in range(i + 1, n_bicliques):
                    if time.time() > deadline - 10:
                        break
                    lj, rj = best[j][0], best[j][1]
                    
                    # Strategy: Check if (li ∪ lj, ri ∪ rj) forms a valid biclique
                    # whose cross-edge set is exactly the union of the two bicliques' edges
                    new_left = set(li) | set(lj)
                    new_right = set(ri) | set(rj)
                    
                    # Quick check: no vertex appears on both sides
                    if new_left & new_right:
                        continue
                    
                    # Check if all cross pairs are edges
                    valid_merge = True
                    for u in new_left:
                        for v in new_right:
                            if tuple(sorted((u, v))) not in edges_set:
                                valid_merge = False
                                break
                        if not valid_merge:
                            break
                    
                    if not valid_merge:
                        continue
                    
                    # Check coverage: merged edges must equal union of both bicliques' edges
                    covered_i = set()
                    for u in li:
                        for v in ri:
                            covered_i.add(tuple(sorted((u, v))))
                    covered_j = set()
                    for u in lj:
                        for v in rj:
                            covered_j.add(tuple(sorted((u, v))))
                    
                    covered_new = set()
                    for u in new_left:
                        for v in new_right:
                            covered_new.add(tuple(sorted((u, v))))
                    
                    if covered_new == covered_i | covered_j:
                        best[i] = [sorted(new_left), sorted(new_right)]
                        del best[j]
                        improved = True
                        break
                if improved:
                    break
        
        # Re-validate after merges
        if _validate_bicliques(best, edges_set):
            if len(best) < best_count:
                best_count = len(best)
        else:
            best = star_partition()
    
    # Polish pass: try to absorb small bicliques into larger ones
    if time.time() < deadline - 5:
        polish_improved = True
        while polish_improved and time.time() < deadline - 5:
            polish_improved = False
            n_b = len(best)
            for i in range(n_b):
                if time.time() > deadline - 5:
                    break
                li, ri = best[i][0], best[i][1]
                for j in range(n_b):
                    if i == j or time.time() > deadline - 5:
                        continue
                    lj, rj = best[j][0], best[j][1]
                    
                    # Try adding some vertices from biclique j to biclique i
                    # Try adding a vertex from lj to li (if it connects to all of ri)
                    for u in lj:
                        if u not in li and all(tuple(sorted((u, v))) in edges_set for v in ri):
                            # Check that this doesn't create duplicate edges
                            new_li = list(li) + [u]
                            new_covered = set()
                            for a in new_li:
                                for b in ri:
                                    new_covered.add(tuple(sorted((a, b))))
                            # The new edges must all be in edges_set and not already covered by other bicliques
                            # For simplicity, just check they're valid edges
                            if all(e in edges_set for e in new_covered):
                                # Verify full partition still works
                                test_best = [list(x) for x in best]
                                test_best[i] = [new_li, ri]
                                test_best[j] = [[u2 for u2 in lj if u2 != u], rj]
                                if test_best[j][0]:  # non-empty
                                    if _validate_bicliques(test_best, edges_set):
                                        best = test_best
                                        polish_improved = True
                                        break
                                break
                    if polish_improved:
                        break
                if polish_improved:
                    break
    
    # Ensure types are correct
    result = []
    for bic in best:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    
    return result

# EVOLVE-BLOCK-END
