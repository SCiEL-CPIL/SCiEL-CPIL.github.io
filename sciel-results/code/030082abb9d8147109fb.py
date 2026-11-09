"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    deadline = time.time() + 55.0  # safety margin before 60s
    n = P['n']
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((u, v))
        edge_set.add((v, u))

    def bfs_shortest(s, t, blocked=None):
        """Find shortest path from s to t avoiding blocked vertices (except endpoints)."""
        if blocked is None:
            blocked = set()
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        q = deque([s])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb in blocked and nb != t:
                    continue
                visited.add(nb)
                parent[nb] = cur
                if nb == t:
                    # reconstruct
                    path = [t]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(nb)
        return None

    def find_disjoint_paths(branches):
        """
        For a given set of branch vertices, try to find pairwise internally-disjoint paths
        for every unordered pair. Returns dict of (u,v) -> path, or None if impossible.
        We use a greedy approach: for each pair, find a path avoiding all previously used
        interior vertices and all branch vertices (except endpoints).
        """
        k = len(branches)
        branch_set = set(branches)
        used_interiors = set()
        paths = {}
        # Order pairs to minimize conflicts: process pairs with shorter natural distance first
        pairs = list(combinations(branches, 2))
        # Sort by BFS distance (shorter first)
        def pair_dist(p):
            path = bfs_shortest(p[0], p[1], used_interiors | branch_set)
            return len(path) if path else 10**9
        pairs.sort(key=pair_dist)
        for u, v in pairs:
            # Try to find a path from u to v avoiding used_interiors and all branches except u,v
            blocked = used_interiors | (branch_set - {u, v})
            path = bfs_shortest(u, v, blocked)
            if path is None:
                return None
            for mid in path[1:-1]:
                used_interiors.add(mid)
            paths[(u, v)] = path
        return paths

    def try_local_search(branches):
        """Try to add vertices one at a time, checking feasibility."""
        current = list(branches)
        improved = True
        while improved:
            improved = False
            for cand in range(n):
                if cand in current:
                    continue
                new_set = current + [cand]
                if find_disjoint_paths(new_set) is not None:
                    current = new_set
                    improved = True
                    break
        return current

    def try_remove_and_readd(branches):
        """Local improvement: remove a vertex, try to re-add a different one."""
        current = list(branches)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                removed = current[i]
                reduced = current[:i] + current[i+1:]
                # Try to add any vertex not in reduced
                for cand in range(n):
                    if cand in reduced:
                        continue
                    new_set = reduced + [cand]
                    if find_disjoint_paths(new_set) is not None:
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
        return current

    # Start with a small seed: find a triangle or just 2 vertices
    best_branches = None
    best_paths = None

    # Try to find a good initial set greedily
    # Start with the two most connected vertices
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])

    # Greedy: start with top-2, try to grow
    initial = [order[0], order[1]]
    # Verify
    p = find_disjoint_paths(initial)
    if p is not None:
        best_branches = initial
        best_paths = p

    # Try to grow from various seeds
    for start_size in range(2, 5):
        if time.time() > deadline - 5:
            break
        seed = order[:start_size]
        p = find_disjoint_paths(seed)
        if p is None:
            continue
        grown = try_local_search(seed)
        if len(grown) > len(best_branches or []):
            p2 = find_disjoint_paths(grown)
            if p2 is not None:
                best_branches = grown
                best_paths = p2

    # Local improvement on best
    if best_branches and time.time() < deadline - 10:
        improved = try_remove_and_readd(best_branches)
        if len(improved) > len(best_branches):
            p = find_disjoint_paths(improved)
            if p is not None:
                best_branches = improved
                best_paths = p

    # Final verification and cleanup
    if best_branches is None or len(best_branches) < 2:
        # Fallback: just use first edge
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # Re-verify the final solution
    final_paths = find_disjoint_paths(best_branches)
    if final_paths is None:
        # Fallback
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # Build output
    branches_out = [int(b) for b in best_branches]
    paths_out = []
    for u, v in combinations(best_branches, 2):
        path = final_paths[(u, v)]
        paths_out.append([int(x) for x in path])

    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
