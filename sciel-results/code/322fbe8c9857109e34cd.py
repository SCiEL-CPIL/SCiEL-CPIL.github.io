import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 58.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def find_path(src, dst, forbidden, max_len=10):
        """Find a path from src to dst avoiding forbidden, preferring shorter paths."""
        if src == dst:
            return None
        if dst in forbidden:
            return None
        # BFS for shortest path
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            if len(path) > max_len:
                continue
            for nb in sorted(adj[node]):
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def find_path_excluding(src, dst, forbidden, exclude_interior):
        """Find a path from src to dst avoiding forbidden and exclude_interior."""
        if src == dst:
            return None
        if dst in forbidden or dst in exclude_interior:
            return None
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            if len(path) > 12:
                continue
            for nb in sorted(adj[node]):
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in exclude_interior and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def try_add_branch(branches, used_interiors, candidate):
        """Try to add candidate as a new branch. Returns (success, new_used_interiors) or (False, None)."""
        branch_set = set(branches) | {candidate}
        new_used = set(used_interiors)
        # Check all pairs involving candidate
        for b in branches:
            forbidden = branch_set - {b, candidate}
            path = find_path(b, candidate, forbidden)
            if path is None:
                return (False, None)
            interior = set(path[1:-1])
            if interior & new_used:
                return (False, None)
            if interior & branch_set:
                return (False, None)
            new_used.update(interior)
        return (True, new_used)

    def solve_with_branches(branches):
        """Given a fixed set of branches, find internally disjoint paths for all pairs."""
        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        paths = {}
        used_interior = set()

        def backtrack(idx):
            if idx == len(pairs):
                return True
            u, v = pairs[idx]
            forbidden = branch_set - {u, v} | used_interior
            path = find_path(u, v, forbidden)
            if path is None:
                return False
            interior = set(path[1:-1])
            if interior & used_interior:
                return False
            if interior & branch_set:
                return False
            paths[(u, v)] = path
            used_interior.update(interior)
            if backtrack(idx + 1):
                return True
            used_interior -= interior
            del paths[(u, v)]
            return False

        if backtrack(0):
            return paths
        return None

    # Strategy: Build branch set incrementally using greedy + backtracking
    # Start with promising initial sets and grow them

    best_count = 0
    best_result = None

    def update_best(branches, paths):
        nonlocal best_count, best_result
        if len(branches) > best_count:
            best_count = len(branches)
            sorted_b = sorted(branches)
            result_paths = []
            for u, v in combinations(sorted_b, 2):
                key = (min(u,v), max(u,v))
                result_paths.append([int(x) for x in paths[key]])
            best_result = {'branches': [int(x) for x in sorted_b], 'paths': result_paths}

    # Phase 1: Try all pairs of high-degree vertices as seeds, then grow
    deg_sorted = sorted(range(n), key=lambda v: -len(adj[v]))

    # Try growing from each pair of top vertices
    for i in range(min(8, len(deg_sorted))):
        for j in range(i+1, min(8, len(deg_sorted))):
            if time.time() > deadline:
                break
            seed = [deg_sorted[i], deg_sorted[j]]
            # Check if path exists
            path = find_path(seed[0], seed[1], set())
            if path is None:
                continue
            used = set(path[1:-1])
            branches = list(seed)
            # Greedily add vertices
            for candidate in deg_sorted:
                if time.time() > deadline:
                    break
                if candidate in branches:
                    continue
                ok, new_used = try_add_branch(branches, used, candidate)
                if ok:
                    branches.append(candidate)
                    used = new_used
            # Try to solve
            paths = solve_with_branches(branches)
            if paths is not None:
                update_best(branches, paths)

    # Phase 2: Try all triples of top vertices
    for combo in combinations(deg_sorted[:10], 3):
        if time.time() > deadline:
            break
        branches = list(combo)
        # Check all pairs have paths
        branch_set = set(branches)
        used = set()
        valid = True
        for u, v in combinations(branches, 2):
            path = find_path(u, v, branch_set - {u, v} | used)
            if path is None:
                valid = False
                break
            interior = set(path[1:-1])
            if interior & used or interior & branch_set:
                valid = False
                break
            used.update(interior)
        if not valid:
            continue
        # Grow
        for candidate in deg_sorted:
            if time.time() > deadline:
                break
            if candidate in branches:
                continue
            ok, new_used = try_add_branch(branches, used, candidate)
            if ok:
                branches.append(candidate)
                used = new_used
        paths = solve_with_branches(branches)
        if paths is not None:
            update_best(branches, paths)

    # Phase 3: Systematic search - try to find larger sets
    # Use a different ordering: try vertices in various orders
    import random
    random.seed(123)

    for trial in range(200):
        if time.time() > deadline:
            break
        # Random subset of size 4-6 from top 15 vertices
        size = random.randint(4, 6)
        pool = deg_sorted[:15]
        subset = random.sample(pool, min(size, len(pool)))
        branches = list(subset)
        branch_set = set(branches)
        used = set()
        valid = True
        for u, v in combinations(branches, 2):
            path = find_path(u, v, branch_set - {u, v} | used)
            if path is None:
                valid = False
                break
            interior = set(path[1:-1])
            if interior & used or interior & branch_set:
                valid = False
                break
            used.update(interior)
        if not valid:
            continue
        # Grow
        for candidate in deg_sorted:
            if time.time() > deadline:
                break
            if candidate in branches:
                continue
            ok, new_used = try_add_branch(branches, used, candidate)
            if ok:
                branches.append(candidate)
                used = new_used
        paths = solve_with_branches(branches)
        if paths is not None:
            update_best(branches, paths)

    # Phase 4: Try to improve best by adding one more vertex
    if best_result is not None and time.time() < deadline - 5:
        current_branches = list(best_result['branches'])
        for candidate in range(n):
            if time.time() > deadline:
                break
            if candidate in current_branches:
                continue
            new_branches = current_branches + [candidate]
            paths = solve_with_branches(new_branches)
            if paths is not None:
                update_best(new_branches, paths)
                current_branches = new_branches

    # Fallback
    if best_result is not None:
        return best_result

    for u, v in P['edges']:
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    return {'branches': [0, 1], 'paths': [[0, 1]]}
