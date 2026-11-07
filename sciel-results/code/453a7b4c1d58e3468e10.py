"""Rewritten program with a stronger clique-subdivision solver."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Helper: check if a set/list of vertices is a clique
    def is_clique(vertices):
        verts = list(vertices)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] not in adj[verts[i]]:
                    return False
        return True

    # Helper: find a simple path between u and v avoiding forbidden vertices (BFS)
    def find_path_avoiding(u, v, forbidden):
        if u == v:
            return [u]
        if v in forbidden or u in forbidden:
            return None
        queue = deque()
        queue.append((u, [u]))
        visited = {u}
        while queue:
            node, path = queue.popleft()
            if node == v:
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # Helper: find ALL simple paths of length <= max_len between u and v avoiding forbidden
    def find_short_paths(u, v, forbidden, max_len=4):
        """Returns list of paths (each a list of vertices) from u to v,
        each of length <= max_len (number of edges), avoiding forbidden in interiors."""
        if u == v:
            return [[u]]
        if v in forbidden or u in forbidden:
            return []
        results = []
        # DFS with depth limit
        def dfs(node, path, depth):
            if len(results) > 200:  # cap
                return
            if node == v and depth > 0:
                results.append(list(path))
                return
            if depth >= max_len:
                return
            for neighbor in adj[node]:
                if neighbor == u and len(path) > 1:
                    continue  # don't revisit start
                if neighbor in path:
                    continue  # simple path
                if neighbor in forbidden:
                    continue
                path.append(neighbor)
                dfs(neighbor, path, depth + 1)
                path.pop()
        dfs(u, [u], 0)
        return results

    # Helper: Validate a full clique subdivision solution
    def validate_solution(branches, paths_dict):
        if len(branches) != len(set(branches)):
            return False
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                pair = (branches[i], branches[j])
                if pair not in paths_dict and (branches[j], branches[i]) not in paths_dict:
                    return False
        all_interiors = set()
        for pair, path in paths_dict.items():
            u, v = pair
            if path[0] != u and path[0] != v:
                return False
            if path[-1] != u and path[-1] != v:
                return False
            if len(path) != len(set(path)):
                return False
            for k in range(len(path) - 1):
                if path[k + 1] not in adj[path[k]]:
                    return False
            interiors = set(path[1:-1])
            for b in branches:
                if b in interiors:
                    return False
            if all_interiors & interiors:
                return False
            all_interiors.update(interiors)
        return True

    # =====================================================================
    # STRATEGY:
    # 1. Find maximum clique (up to size 7) using Bron-Kerbosch style or brute force.
    # 2. If clique size >= 5, try to EXTEND it by adding vertices via subdivision paths.
    # 3. Also try to find K5, K6 subdivisions directly.
    # 4. Use a smarter approach: start with a clique and greedily add vertices.
    # =====================================================================

    # Step 1: Find all cliques of size 4, 5, 6, 7 (max feasible)
    # Use a simple approach: for each vertex, find maximal cliques in its neighborhood.

    # Actually, let's use a different structural approach:
    # Find the maximum clique first, then try to grow it.

    def find_max_clique():
        """Find a maximum clique using branch and bound."""
        best_clique = []

        def branch(candidates, current_clique, depth):
            nonlocal best_clique
            if time.time() - start_time > DEADLINE - 5:
                return
            if not candidates:
                if len(current_clique) > len(best_clique):
                    best_clique = list(current_clique)
                return
            # Upper bound: if even adding all candidates can't beat best, prune
            if len(current_clique) + len(candidates) <= len(best_clique):
                return

            # Try each candidate
            for i, v in enumerate(candidates):
                # Check v is adjacent to all in current_clique
                if all(v in adj[c] for c in current_clique):
                    current_clique.append(v)
                    # New candidates: those after i that are adjacent to v
                    new_candidates = [candidates[j] for j in range(i + 1, len(candidates))
                                      if v in adj[candidates[j]]]
                    branch(new_candidates, current_clique, depth + 1)
                    current_clique.pop()

        branch(list(range(n)), [], 0)
        return best_clique

    max_clique = find_max_clique()
    max_clique_size = len(max_clique)

    # If we found a clique of size >= 4, we can use it directly
    if max_clique_size >= 4:
        branches = list(max_clique)
        paths_list = []
        for pair in itertools.combinations(branches, 2):
            paths_list.append(list(pair))
        return {'branches': branches, 'paths': paths_list}

    # Step 2: Try to find a K4 or larger subdivision
    # Key insight: we need pairwise internally-disjoint paths between all pairs of branches.

    # Approach: Try all 4-tuples and 5-tuples, use a more sophisticated path-finding
    # that considers all pairs simultaneously.

    def try_find_subdivision(branch_set, max_time=2.0):
        """Try to find internally-disjoint paths for all pairs in branch_set.
        Returns dict of (u,v) -> path, or None."""
        branch_list = list(branch_set)
        k = len(branch_list)
        branch_set = set(branch_list)
        pairs = list(itertools.combinations(branch_list, 2))

        if k == 2:
            # Just need one path
            p = find_path_avoiding(branch_list[0], branch_list[1], set())
            if p is None:
                return None
            return {(branch_list[0], branch_list[1]): p}

        # For k >= 3, we need to find disjoint paths for all pairs.
        # Strategy: For each pair, find candidate paths. Then use backtracking
        # to assign disjoint paths.

        # First, find candidate paths for each pair
        candidate_paths = {}
        for u, v in pairs:
            # Find paths of length 1 (edge), then length 2, then longer
            paths = find_short_paths(u, v, branch_set, max_len=5)
            if not paths:
                # Try BFS for any path
                p = find_path_avoiding(u, v, branch_set)
                if p is not None:
                    paths = [p]
            if not paths:
                return None
            # Sort by length (prefer shorter paths)
            paths.sort(key=len)
            candidate_paths[(u, v)] = paths[:30]  # keep top 30 candidates

        # Now use backtracking to assign disjoint paths
        # Order pairs by number of candidates (most constrained first)
        pair_order = sorted(pairs, key=lambda p: len(candidate_paths[p]))

        def backtrack(idx, used_vertices, assignment):
            if time.time() - start_time > DEADLINE - 3:
                return None
            if idx == len(pair_order):
                return assignment

            u, v = pair_order[idx]
            for path in candidate_paths[(u, v)]:
                interiors = set(path[1:-1])
                if interiors & used_vertices:
                    continue
                # Also check interiors don't contain branches (shouldn't, but safety)
                if interiors & branch_set:
                    continue
                new_used = used_vertices | interiors
                new_assignment = dict(assignment)
                new_assignment[(u, v)] = path
                result = backtrack(idx + 1, new_used, new_assignment)
                if result is not None:
                    return result
            return None

        result = backtrack(0, set(), {})
        return result

    # Try to find K4 subdivisions
    best_result = None
    best_size = 0

    # Try 5-tuples first (for a potential jump)
    # C(24,5) = 42504 - might be too many. Let's be selective.
    # Priority: 5-tuples where at least 8 of the 10 pairs are edges (near-cliques)

    def edge_count_in_set(vertices):
        verts = list(vertices)
        count = 0
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] in adj[verts[i]]:
                    count += 1
        return count

    # Generate candidate sets sorted by edge density
    # For k=5: C(24,5) = 42504
    # For k=4: C(24,4) = 10626

    # Let's try k=5 first with a time limit
    found_k5 = False
    for comb in itertools.combinations(range(n), 5):
        if time.time() - start_time > DEADLINE - 15:
            break

        # Quick filter: need at least some edges
        ec = edge_count_in_set(comb)
        if ec < 8:  # at least 8 of 10 pairs must be edges
            continue

        result = try_find_subdivision(set(comb), max_time=1.0)
        if result is not None:
            branches = list(comb)
            paths_list = []
            for pair in itertools.combinations(branches, 2):
                p = result.get(pair) or result.get((pair[1], pair[0]))
                if p is None:
                    result = None
                    break
                paths_list.append(p)
            if result is not None and validate_solution(branches, result):
                return {'branches': branches, 'paths': paths_list}

    # Try k=4
    for comb in itertools.combinations(range(n), 4):
        if time.time() - start_time > DEADLINE - 5:
            break

        ec = edge_count_in_set(comb)
        if ec < 5:  # at least 5 of 6 pairs must be edges
            continue

        result = try_find_subdivision(set(comb), max_time=0.5)
        if result is not None:
            branches = list(comb)
            paths_list = []
            for pair in itertools.combinations(branches, 2):
                p = result.get(pair) or result.get((pair[1], pair[0]))
                if p is None:
                    result = None
                    break
                paths_list.append(p)
            if result is not None and validate_solution(branches, result):
                if best_size < 4:
                    best_size = 4
                    best_result = {'branches': branches, 'paths': paths_list}
                # Don't break - might find a better one? No, all k=4 are equal.
                # But let's keep searching for k=5 if time allows.
                # Actually, let's return the first valid k=4 we find.
                return {'branches': branches, 'paths': paths_list}

    # Fallback: find any triangle (K3)
    for comb in itertools.combinations(range(n), 3):
        if is_clique(comb):
            branches = list(comb)
            paths_list = []
            for pair in itertools.combinations(branches, 2):
                paths_list.append(list(pair))
            return {'branches': branches, 'paths': paths_list}

    # Fallback: find any K3 subdivision
    for comb in itertools.combinations(range(n), 3):
        if time.time() - start_time > DEADLINE - 2:
            break
        result = try_find_subdivision(set(comb), max_time=0.3)
        if result is not None:
            branches = list(comb)
            paths_list = []
            for pair in itertools.combinations(branches, 2):
                p = result.get(pair) or result.get((pair[1], pair[0]))
                if p is None:
                    result = None
                    break
                paths_list.append(p)
            if result is not None and validate_solution(branches, result):
                return {'branches': branches, 'paths': paths_list}

    # Final fallback: edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
