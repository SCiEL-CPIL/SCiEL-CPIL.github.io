"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations

    t_start = time.time()
    DEADLINE = 82.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_path_bfs(src, dst, forbidden_interior):
        """BFS from src to dst, interior vertices avoid forbidden_interior."""
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
            for nb in sorted(adj[cur]):
                if nb not in parent:
                    if nb != dst and nb in forbidden_interior:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_path_bfs_reverse(src, dst, forbidden_interior):
        """BFS from dst to src (reverse), interior vertices avoid forbidden_interior."""
        if src == dst:
            return [src]
        parent = {dst: None}
        queue = deque([dst])
        while queue:
            cur = queue.popleft()
            if cur == src:
                path = []
                node = src
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in sorted(adj[cur]):
                if nb not in parent:
                    if nb != src and nb in forbidden_interior:
                        continue
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_path_bfs_both(src, dst, forbidden_interior):
        """Try BFS in both directions, return shorter path."""
        p1 = find_path_bfs(src, dst, forbidden_interior)
        p2 = find_path_bfs_reverse(src, dst, forbidden_interior)
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        if len(p1) <= len(p2):
            return p1
        return p2

    def try_branch_set(branches, max_retries=3):
        """Try to find internally-disjoint paths for all pairs.
        Uses multiple pair orderings and path-finding strategies."""
        bset = set(branches)
        k = len(branches)
        pairs = list(combinations(branches, 2))

        # Generate multiple orderings
        orderings = []
        # Ordering 1: direct edges first, then by distance
        pairs_sorted1 = sorted(pairs, key=lambda p: (0 if p[1] in adj[p[0]] else 1, p[0], p[1]))
        orderings.append(pairs_sorted1)
        # Ordering 2: reverse
        orderings.append(list(reversed(pairs_sorted1)))
        # Ordering 3: by sum of degrees descending
        deg = {v: len(adj[v]) for v in branches}
        pairs_sorted3 = sorted(pairs, key=lambda p: -(deg[p[0]] + deg[p[1]]))
        orderings.append(pairs_sorted3)
        # Ordering 4: by min degree ascending (harder pairs first)
        pairs_sorted4 = sorted(pairs, key=lambda p: (deg[p[0]], deg[p[1]]))
        orderings.append(pairs_sorted4)

        for ordering in orderings:
            if time_up():
                return None
            paths = {}
            used_interiors = set()
            success = True

            for u, v in ordering:
                if time_up():
                    success = False
                    break
                # Try multiple path-finding strategies
                path = None
                path = find_path_bfs_both(u, v, used_interiors)
                if path is None:
                    # Try with slightly different approach: BFS from v
                    path = find_path_bfs(v, u, used_interiors)
                if path is None:
                    success = False
                    break

                interior = set(path[1:-1])
                if interior & bset:
                    success = False
                    break
                if interior & used_interiors:
                    success = False
                    break
                used_interiors |= interior
                paths[(u, v)] = path

            if success:
                path_list = []
                for u, v in combinations(branches, 2):
                    path_list.append(paths[(u, v)])
                return {'branches': list(branches), 'paths': path_list}

        return None

    def try_branch_set_backtrack(branches):
        """Backtracking approach: try to assign paths one by one,
        with backtracking on failure."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort: direct edges first
        pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1, p[0], p[1]))

        def backtrack(idx, used_interiors, paths):
            if time_up():
                return None
            if idx == len(pairs):
                path_list = []
                for u, v in combinations(branches, 2):
                    path_list.append(paths[(u, v)])
                return {'branches': list(branches), 'paths': path_list}

            u, v = pairs[idx]
            # Try BFS
            path = find_path_bfs_both(u, v, used_interiors)
            if path is not None:
                interior = set(path[1:-1])
                if not (interior & bset) and not (interior & used_interiors):
                    used_interiors.add  # copy
                    new_used = used_interiors | interior
                    paths[(u, v)] = path
                    result = backtrack(idx + 1, new_used, paths)
                    if result is not None:
                        return result
                    del paths[(u, v)]

            return None

        return backtrack(0, set(), {})

    # Degree-based candidate ranking
    deg = [len(adj[i]) for i in range(n)]
    candidates = sorted(range(n), key=lambda i: (-deg[i], i))

    best = None
    best_size = 0

    # Try sizes from large to small
    for size in range(min(n, 12), 1, -1):
        if time_up():
            break
        if size < best_size:
            break

        # For each size, try combinations of top candidates
        # Use top (size+3) candidates to keep search manageable
        top_k = min(n, size + 4)
        tried = 0
        max_tries = 3000

        for combo in combinations(candidates[:top_k], size):
            if time_up() or tried >= max_tries:
                break
            tried += 1
            branches = list(combo)
            result = try_branch_set(branches)
            if result is not None:
                if size > best_size:
                    best = result
                    best_size = size
                break

        if best_size >= size:
            break

    # If we didn't find a large one, try with backtracking on smaller sets
    if best is None or best_size < 7:
        for size in range(min(n, 9), max(best_size, 3), -1):
            if time_up():
                break
            if size <= best_size:
                break
            top_k = min(n, size + 3)
            tried = 0
            max_tries = 2000
            for combo in combinations(candidates[:top_k], size):
                if time_up() or tried >= max_tries:
                    break
                tried += 1
                branches = list(combo)
                result = try_branch_set_backtrack(branches)
                if result is not None:
                    if size > best_size:
                        best = result
                        best_size = size
                    break
            if best_size >= size:
                break

    # Fallback: find a triangle
    if best is None:
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
