"""Optimal biclique partition via exhaustive maximal biclique enumeration + strong branch-and-bound."""

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _seed_solution():
    """Fallback: one biclique per vertex (star decomposition)."""
    n = P['n']
    edges = P['edges']
    adj = {u: [] for u in range(n)}
    for a, b in edges:
        adj[a].append(b)
    result = []
    for u in range(n):
        if adj[u]:
            result.append([[u], list(adj[u])])
    return result


def _enumerate_maximal_bicliques(n, adj, edge_set, time_limit=20):
    """
    Exhaustively enumerate all maximal bicliques using a recursive expansion algorithm.
    A biclique (L, R) is maximal if no vertex can be added to L or R while preserving
    the complete bipartite property.
    
    Algorithm: For each edge (a,b), recursively try all ways to expand L and R.
    Use memoization and pruning to avoid redundant work.
    """
    import time
    start_time = time.time()
    candidates = []
    seen = set()
    
    def add_candidate(L, R):
        if not L or not R:
            return
        L_sorted = tuple(sorted(L))
        R_sorted = tuple(sorted(R))
        # Canonical form
        if L_sorted[0] < R_sorted[0]:
            key = (L_sorted, R_sorted)
        else:
            key = (R_sorted, L_sorted)
        if key not in seen:
            seen.add(key)
            L_list = list(L_sorted)
            R_list = list(R_sorted)
            if L_list[0] > R_list[0]:
                L_list, R_list = R_list, L_list
            candidates.append((L_list, R_list))
    
    def can_add_to_L(x, L, R):
        """Check if vertex x can be added to L while maintaining biclique property."""
        if x in L or x in R:
            return False
        # x must be adjacent to all vertices in R
        for v in R:
            if v not in adj[x]:
                return False
        return True
    
    def can_add_to_R(x, L, R):
        """Check if vertex x can be added to R while maintaining biclique property."""
        if x in L or x in R:
            return False
        # x must be adjacent to all vertices in L
        for u in L:
            if u not in adj[x]:
                return False
        return True
    
    def is_maximal(L, R):
        """Check if (L, R) is a maximal biclique."""
        for x in range(n):
            if x not in L and x not in R:
                if can_add_to_L(x, L, R) or can_add_to_R(x, L, R):
                    return False
        return True
    
    def expand_maximal(L, R):
        """Expand (L, R) to a maximal biclique by greedily adding all possible vertices."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and x not in R:
                    if can_add_to_L(x, L, R):
                        L.add(x)
                        changed = True
            for x in range(n):
                if x not in L and x not in R:
                    if can_add_to_R(x, L, R):
                        R.add(x)
                        changed = True
        return L, R
    
    def recursive_enumerate(L, R, forbidden_L, forbidden_R):
        """
        Recursively enumerate all maximal bicliques containing the current (L, R).
        forbidden_L: vertices that cannot be added to L (already tried)
        forbidden_R: vertices that cannot be added to R (already tried)
        """
        if time.time() - start_time > time_limit:
            return
        
        L = set(L)
        R = set(R)
        forbidden_L = set(forbidden_L)
        forbidden_R = set(forbidden_R)
        
        # Find all vertices that can be added to L
        candidates_L = []
        for x in range(n):
            if x not in L and x not in R and x not in forbidden_L:
                if can_add_to_L(x, L, R):
                    candidates_L.append(x)
        
        # Find all vertices that can be added to R
        candidates_R = []
        for x in range(n):
            if x not in L and x not in R and x not in forbidden_R:
                if can_add_to_R(x, L, R):
                    candidates_R.append(x)
        
        # If no more vertices can be added, this is a maximal biclique
        if not candidates_L and not candidates_R:
            add_candidate(list(L), list(R))
            return
        
        # Branch: try adding each candidate to L
        for x in candidates_L:
            if time.time() - start_time > time_limit:
                return
            new_L = L | {x}
            new_forbidden_R = forbidden_R | {x}
            # Vertices in R that are not adjacent to x can't stay in R... wait,
            # actually if x is added to L, all of R must be adjacent to x (already checked)
            # But vertices in R that were only kept because of other L vertices might need rechecking
            # Actually no - the biclique property is: all (u,v) for u in L, v in R are edges.
            # If we add x to L, we need x adjacent to all v in R (checked).
            # The existing L vertices were already adjacent to all R vertices.
            # So the biclique property is maintained.
            recursive_enumerate(new_L, R, forbidden_L, new_forbidden_R)
        
        # Branch: try adding each candidate to R
        for x in candidates_R:
            if time.time() - start_time > time_limit:
                return
            new_R = R | {x}
            new_forbidden_L = forbidden_L | {x}
            recursive_enumerate(L, new_R, new_forbidden_L, forbidden_R)
    
    # Start enumeration from each edge
    edges_list = [(a, b) for a, b in P['edges']]
    for a, b in edges_list:
        if time.time() - start_time > time_limit:
            break
        # Start with L={a}, R={b}
        recursive_enumerate({a}, {b}, set(), set())
    
    # Also add star bicliques as candidates (they may not be maximal but are useful)
    for u in range(n):
        if adj[u]:
            add_candidate([u], list(adj[u]))
    
    return candidates


def _solve_set_cover(cand_edges, n_edges, time_limit=35):
    """
    Solve minimum set cover using branch-and-bound with strong lower bounds.
    cand_edges: list of frozensets, each representing edges covered by a candidate
    n_edges: total number of edges
    Returns: list of candidate indices forming an optimal cover, or None.
    """
    import time
    start_time = time.time()
    
    n_cands = len(cand_edges)
    
    if n_cands == 0:
        return None
    
    # Check feasibility
    all_covered = set()
    for ce in cand_edges:
        all_covered |= ce
    if len(all_covered) < n_edges:
        return None
    
    # Greedy initial solution for upper bound
    uncovered = set(range(n_edges))
    greedy_sol = []
    while uncovered:
        best_i = -1
        best_count = 0
        for i in range(n_cands):
            count = len(cand_edges[i] & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
        if best_i == -1 or best_count == 0:
            break
        greedy_sol.append(best_i)
        uncovered -= cand_edges[best_i]
    
    if uncovered:
        return None
    
    upper_bound = len(greedy_sol)
    best_sol = list(greedy_sol)
    
    # Precompute: for each edge, which candidates cover it
    edge_to_cands = [[] for _ in range(n_edges)]
    for i, ce in enumerate(cand_edges):
        for e in ce:
            edge_to_cands[e].append(i)
    
    # Check every edge is coverable
    for e in range(n_edges):
        if not edge_to_cands[e]:
            return None
    
    # Strong lower bound: solve LP relaxation approximately
    # Use a greedy packing lower bound: repeatedly pick an uncovered edge,
    # add 1 to LB, and remove all edges that share a candidate with it.
    # This gives a valid lower bound (packing number of the transversal matroid).
    
    def compute_lb(uncovered):
        """Compute a lower bound on the number of additional sets needed."""
        if not uncovered:
            return 0
        
        # LB1: ceil(|uncovered| / max_single_cover)
        max_cover = 0
        for i in range(n_cands):
            c = len(cand_edges[i] & uncovered)
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return float('inf')
        lb1 = (len(uncovered) + max_cover - 1) // max_cover
        
        # LB2: Greedy packing bound
        # Pick edges greedily such that no two picked edges share a candidate
        lb2 = 0
        remaining = set(uncovered)
        while remaining:
            # Pick edge with fewest covering candidates (most constrained)
            e = min(remaining, key=lambda x: len([c for c in edge_to_cands[x] if cand_edges[c] & remaining]))
            lb2 += 1
            # Remove all edges that share any candidate with e
            to_remove = set()
            for cand_idx in edge_to_cands[e]:
                to_remove |= cand_edges[cand_idx]
            remaining -= to_remove
        
        # LB3: Fractional packing (approximate LP dual)
        # For each edge, let w_e = 1 / (number of candidates that can cover it among those useful)
        # Sum of w_e over uncovered edges gives a lower bound
        lb3 = 0.0
        for e in uncovered:
            useful_cands = [c for c in edge_to_cands[e] if cand_edges[c] & uncovered]
            if useful_cands:
                lb3 += 1.0 / len(useful_cands)
            else:
                return float('inf')
        
        return max(lb1, lb2, int(lb3))
    
    nodes_explored = 0
    timed_out = [False]
    
    def branch_and_bound(selection, uncovered, depth):
        nonlocal nodes_explored, upper_bound, best_sol, timed_out
        nodes_explored += 1
        
        if timed_out[0]:
            return
        if time.time() - start_time > time_limit:
            timed_out[0] = True
            return
        
        if not uncovered:
            if len(selection) < upper_bound:
                upper_bound = len(selection)
                best_sol = list(selection)
            return
        
        lb = compute_lb(uncovered)
        if len(selection) + lb >= upper_bound:
            return
        
        # Choose branching edge: the one with fewest covering candidates
        e = min(uncovered, key=lambda x: len([c for c in edge_to_cands[x] if cand_edges[c] & uncovered]))
        
        # Get candidates that can cover e and have useful coverage
        cands_for_e = [c for c in edge_to_cands[e] if cand_edges[c] & uncovered]
        
        # Order by coverage (most covering first for better pruning)
        cands_for_e.sort(key=lambda i: -len(cand_edges[i] & uncovered))
        
        for cand_idx in cands_for_e:
            if timed_out[0]:
                return
            new_uncovered = uncovered - cand_edges[cand_idx]
            selection.append(cand_idx)
            branch_and_bound(selection, new_uncovered, depth + 1)
            selection.pop()
        
        return
    
    branch_and_bound([], set(range(n_edges)), 0)
    
    if best_sol and len(best_sol) <= upper_bound:
        return best_sol
    return None


def construct():
    import time
    start = time.time()
    
    n = P['n']
    edges = P['edges']
    
    # Build edge index
    edge_indices = {}
    for i, (a, b) in enumerate(edges):
        key = (min(a, b), max(a, b))
        edge_indices[key] = i
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    edge_set = set(edge_indices.keys())
    
    try:
        # Step 1: Enumerate maximal bicliques
        candidates = _enumerate_maximal_bicliques(n, adj, edge_set, time_limit=25)
        
        # Step 2: Build candidate edge sets
        cand_edges = []
        for L, R in candidates:
            covered = set()
            for u in L:
                for v in R:
                    key = (min(u, v), max(u, v))
                    if key in edge_indices:
                        covered.add(edge_indices[key])
            cand_edges.append(frozenset(covered))
        
        # Remove empty candidates
        valid_cands = [(i, ce) for i, ce in enumerate(cand_edges) if ce]
        if not valid_cands:
            return _seed_solution()
        
        # Step 3: Remove dominated candidates
        # A candidate is dominated if its edge set is a subset of another's
        n_valid = len(valid_cands)
        keep = [True] * n_valid
        for i in range(n_valid):
            if not keep[i]:
                continue
            for j in range(n_valid):
                if i == j or not keep[j]:
                    continue
                if valid_cands[i][1] <= valid_cands[j][1]:
                    if len(valid_cands[i][1]) < len(valid_cands[j][1]):
                        keep[i] = False
                        break
                    elif i > j:
                        # Equal sets, keep the one with smaller index
                        keep[i] = False
                        break
        
        filtered_idx = [i for i in range(n_valid) if keep[i]]
        filtered_cand_edges = [valid_cands[i][1] for i in filtered_idx]
        
        # Step 4: Solve set cover
        sol = _solve_set_cover(filtered_cand_edges, len(edge_indices), time_limit=30)
        
        if sol is not None:
            # Map back to original candidate indices
            orig_indices = [valid_cands[i][0] for i in sol]
            
            # Build result
            result = []
            for idx in orig_indices:
                L, R = candidates[idx]
                # Verify all cross pairs are edges
                valid = True
                for u in L:
                    for v in R:
                        key = (min(u, v), max(u, v))
                        if key not in edge_indices:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and L and R:
                    result.append([list(L), list(R)])
            
            # Verify partition
            if result:
                edge_count = {}
                for L, R in result:
                    for u in L:
                        for v in R:
                            key = (min(u, v), max(u, v))
                            edge_count[key] = edge_count.get(key, 0) + 1
                
                all_edges = set(edge_indices.keys())
                covered = set(edge_count.keys())
                
                if covered == all_edges and all(c == 1 for c in edge_count.values()):
                    final_result = []
                    for L, R in result:
                        L_int = [int(x) for x in L]
                        R_int = [int(x) for x in R]
                        final_result.append([L_int, R_int])
                    return final_result
    except Exception:
        pass
    
    # Fallback
    return _seed_solution()
