"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))

    def find_clique_k(k):
        """Find a clique of size k using backtracking."""
        if k == 1:
            return [0]
        if k == 2:
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        return [u, v]
            return None
        if k > n:
            return None

        # Start with a vertex of high degree
        candidates = list(range(n))
        candidates.sort(key=lambda x: -len(adj[x]))
        # Try to build clique of size k

        def backtrack(current_clique, candidates_for_next):
            if time.time() > deadline:
                return None
            if len(current_clique) == k:
                return list(current_clique)
            # Filter candidates that connect to all in current_clique
            filtered = []
            for c in candidates_for_next:
                if all(c in adj[x] for x in current_clique):
                    filtered.append(c)
            if len(filtered) < k - len(current_clique):
                return None
            # Try each candidate
            for i, c in enumerate(filtered):
                current_clique.append(c)
                result = backtrack(current_clique, filtered[i+1:])
                if result is not None:
                    return result
                current_clique.pop()
            return None

        for seed in candidates:
            if time.time() > deadline:
                return None
            result = backtrack([seed], [c for c in candidates if c != seed])
            if result is not None:
                return result
        return None

    # Try to find the largest clique
    best_clique = None
    for k in range(min(n, 20), 2, -1):
        if time.time() > deadline:
            break
        clique = find_clique_k(k)
        if clique is not None:
            best_clique = clique
            break

    if best_clique is None:
        # Fallback: find a 3-clique
        best_clique = find_clique_k(3)
        if best_clique is None:
            # Fallback: find a 2-clique (edge)
            u, v = edges[0]
            best_clique = [u, v]

    branches = list(best_clique)

    # For each pair of branches, find a simple path whose interior vertices
    # are all non-branch vertices, and such that interiors across different
    # paths are pairwise disjoint.
    # Strategy: for each pair (u, v), find a path in G - (branches \ {u, v}).
    # We need the interiors to be disjoint across all pairs.

    # Build the set of interior vertices available: all vertices not in branches
    interior_available = set(range(n)) - set(branches)

    # We'll find paths greedily. For each pair, do BFS avoiding used interior vertices.
    # Order pairs to make it easier: start with pairs that have direct edges or short paths.

    pairs = list(itertools.combinations(range(len(branches)), 2))

    # Sort pairs by some heuristic: prefer pairs with direct edges first
    def pair_key(p):
        bi, bj = p
        ui, uj = branches[bi], branches[bj]
        # Prefer direct edges
        if (ui, uj) in edge_set or (uj, ui) in edge_set:
            return (0, 0)
        # Prefer pairs with common neighbors
        common = adj[ui] & adj[uj]
        return (1, -len(common))

    pairs.sort(key=pair_key)

    used_interior = set()
    paths_dict = {}

    # For each pair, find a path from branches[bi] to branches[bj]
    # in the graph where we remove all used_interior vertices (but keep all branches)
    # Actually, the path interior must avoid ALL branches except the endpoints.
    # And the interior vertices must be disjoint across all paths.

    for bi, bj in pairs:
        if time.time() > deadline:
            break
        u = branches[bi]
        v = branches[bj]

        # Direct edge?
        if (u, v) in edge_set or (v, u) in edge_set:
            paths_dict[(bi, bj)] = [u, v]
            continue

        # BFS from u to v, avoiding:
        # - all branch vertices except u and v
        # - all used_interior vertices
        # We want a simple path with minimum interior vertices (to leave more available)

        forbidden = set(branches) - {u, v}
        forbidden |= used_interior

        # BFS
        from collections import deque
        parent = {u: None}
        queue = deque([u])
        found = False
        while queue:
            if time.time() > deadline:
                break
            curr = queue.popleft()
            if curr == v:
                found = True
                break
            for nb in adj[curr]:
                if nb in forbidden:
                    continue
                if nb not in parent:
                    parent[nb] = curr
                    queue.append(nb)
        if not found:
            # Try a different approach: allow going through branches?
            # No, the rule says path interiors must avoid every branch.
            # So we can't go through other branches.
            # If no path found, we need to skip this pair or find an alternative.
            # For now, use a direct connection if possible, else skip.
            # Actually, we must include ALL pairs. Let's try a more flexible search.
            # Retry with a different ordering or just use a greedy path.
            # For now, mark as failed and try a simpler approach.
            paths_dict[(bi, bj)] = None
            continue

        # Reconstruct path
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        # Path goes from u to v
        interior = set(path[1:-1])
        # Check that interior doesn't include any branch
        if interior & set(branches):
            # This shouldn't happen if we correctly excluded branches
            paths_dict[(bi, bj)] = None
            continue
        paths_dict[(bi, bj)] = path
        used_interior |= interior

    # Check if all pairs were successfully assigned
    all_ok = all(paths_dict.get((bi, bj)) is not None for bi, bj in pairs)

    if not all_ok:
        # Fallback: try a different approach. Use the original simple method:
        # just use direct edges for all pairs (only works if branches form a clique)
        # If branches don't form a clique, we need to find paths.
        # Let's try a different clique or a different path assignment.
        # For now, fall back to the seed program's approach: find a triangle.
        for row in itertools.combinations(range(n), 3):
            if all(((a, b) in edge_set or (b, a) in edge_set) for a, b in itertools.combinations(row, 2)):
                branches = list(row)
                pairs_fb = list(itertools.combinations(range(len(branches)), 2))
                paths_list = [list(p) for p in pairs_fb]
                # Convert to proper format
                result_paths = []
                for p in pairs_fb:
                    result_paths.append([branches[p[0]], branches[p[1]]])
                return {'branches': branches, 'paths': result_paths}
        # Ultimate fallback
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}

    # Build the final paths list in the order of combinations
    result_paths = []
    for bi, bj in pairs:
        result_paths.append(paths_dict[(bi, bj)])

    return {'branches': branches, 'paths': result_paths}
# EVOLVE-BLOCK-END
