"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations, permutations

    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_path(src, dst, forbidden):
        """Find a simple path from src to dst avoiding vertices in forbidden (interior)."""
        if src == dst:
            return [src]
        if src in forbidden or dst in forbidden:
            return None
        # BFS
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                # reconstruct
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[cur]:
                if nb not in parent and nb not in forbidden:
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_path_avoiding_interior(src, dst, forbidden_interior):
        """Find path from src to dst where interior vertices avoid forbidden_interior.
        Endpoints src,dst are allowed even if in forbidden_interior (they're branches)."""
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[cur]:
                if nb not in parent:
                    # nb is interior unless it's dst
                    if nb != dst and nb in forbidden_interior:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def try_branch_set(branches):
        """Try to find internally-disjoint paths for all pairs of branches.
        Returns dict or None."""
        bset = set(branches)
        # Order pairs: start with edges between branches (direct), then others
        pairs = list(combinations(branches, 2))
        # Sort pairs: direct edges first (shorter, easier)
        pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1, p[0], p[1]))

        paths = {}
        used_interiors = set()

        for u, v in pairs:
            # Find a path from u to v avoiding used_interiors
            # Try BFS
            path = find_path_avoiding_interior(u, v, used_interiors)
            if path is None:
                return None
            # Check interior
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            paths[(u, v)] = path

        # Build result
        path_list = []
        for u, v in combinations(branches, 2):
            path_list.append(paths[(u, v)])

        return {'branches': list(branches), 'paths': path_list}

    # Candidate vertices: high degree first
    deg = [len(adj[i]) for i in range(n)]
    candidates = sorted(range(n), key=lambda i: (-deg[i], i))

    best = None
    best_size = 0

    # Try sizes from large to small
    for size in range(min(n, 12), 1, -1):
        if time_up():
            break
        # Try subsets of candidates of this size
        # Limit number of tries
        tried = 0
        max_tries = 5000
        for combo in combinations(candidates, size):
            if time_up() or tried >= max_tries:
                break
            tried += 1
            branches = list(combo)
            result = try_branch_set(branches)
            if result is not None:
                if size > best_size:
                    best = result
                    best_size = size
                # Found one of this size; try to find larger, so break to next size
                break
        if best_size >= size:
            # Found a solution of this size, look for larger (already going down, so this means we found max)
            # Actually we go from large to small, so if we found one, it's the largest we'll find
            break

    # If nothing found, fall back to triangle
    if best is None:
        # Find any triangle
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            u, v = edges[0]
            best = {'branches': [u, v], 'paths': [[u, v]]}

    # Convert to proper types
    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
