"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

import time
import random
from collections import deque
from itertools import combinations

def _build_graph():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    n, adj = _build_graph()
    adj_list = [sorted(adj[i]) for i in range(n)]
    deg = [len(adj[i]) for i in range(n)]
    adjset = [adj[i] for i in range(n)]

    def time_up():
        return time.time() - t0 > DEADLINE

    # BFS path avoiding forbidden interior vertices (endpoints allowed)
    def bfs_path(src, dst, forbidden):
        if src == dst:
            return [src]
        parent = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj_list[cur]:
                if nb in parent:
                    continue
                if nb != dst and nb in forbidden:
                    continue
                parent[nb] = cur
                q.append(nb)
        return None

    def pair_order(branches):
        # Order pairs: direct edges first, then by a proxy for difficulty
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        def key(p):
            u, v = p
            direct = 0 if v in adjset[u] else 1
            # fewer common neighbors -> harder to route? use sum of degrees
            return (direct, deg[u] + deg[v], u, v)
        pairs.sort(key=key)
        return pairs

    # Try to route all pairs for a given branch set, with backtracking on order.
    def try_route(branches, time_budget=0.25):
        bset = set(branches)
        pairs = pair_order(branches)
        # We'll do a DFS over pair permutations? Too many. Instead:
        # Try the fixed order with backtracking: if a pair fails, try a different
        # path for earlier pairs? That's expensive. Use a simpler strategy:
        # Try multiple random orderings and multiple BFS attempts (BFS is unique
        # shortest, so to vary we can try different forbidden sets).
        # Strategy: iterative - for each ordering attempt, greedily route with BFS.
        # If it fails, we do a limited backtracking: re-route the failed pair by
        # trying to free up space (not implemented fully). Instead, rely on
        # trying many branch sets and orderings.

        # First attempt: fixed order
        used = set()
        paths = {}
        for u, v in pairs:
            p = bfs_path(u, v, used)
            if p is None:
                return None
            interior = set(p[1:-1])
            if interior & bset:
                return None
            if interior & used:
                return None
            used |= interior
            paths[(u, v)] = p
        # success
        return paths

    def try_route_with_retries(branches, n_retries=30):
        """Try the fixed order, then try randomized neighbor orderings to break BFS ties."""
        # First: fixed
        r = try_route(branches)
        if r is not None:
            return r
        # Randomize: temporarily shuffle adjacency lists? BFS with deque uses
        # adjacency order; shuffling gives different shortest paths.
        saved = [list(l) for l in adj_list]
        for _ in range(n_retries):
            if time_up():
                break
            for i in range(n):
                random.shuffle(adj_list[i])
            r = try_route(branches)
            if r is not None:
                for i in range(n):
                    adj_list[i] = saved[i]
                return r
        for i in range(n):
            adj_list[i] = saved[i]
        return None

    # Backtracking router: try to route pairs; on failure, backtrack by
    # choosing a different path for a previously routed pair.
    def try_route_bt(branches, max_nodes=200000):
        bset = set(branches)
        pairs = pair_order(branches)
        m = len(pairs)
        # State: list of paths chosen so far, used set
        # We'll implement a recursive backtracking that, for the current pair,
        # enumerates a few candidate paths (shortest + a few alternatives) and
        # recurses.
        # To keep it bounded, only allow backtracking depth and limited candidates.

        # Precompute candidate paths per pair given current forbidden? That's
        # dynamic. Instead, at each step, find up to K disjoint-ish paths via BFS
        # with different neighbor shuffles.

        # Simpler backtracking: if a pair fails, try re-routing the LAST pair
        # with a different path, etc. We'll do a DFS over "path choice index".
        # Given complexity, implement a limited version.

        # For each pair, generate up to 5 candidate paths (by shuffling neighbors)
        # and DFS.

        def gen_paths(src, dst, forbidden, k=5):
            cands = []
            seen = set()
            # first: fixed order
            p = bfs_path(src, dst, forbidden)
            if p:
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    cands.append(p)
            if len(cands) < k:
                saved = [list(l) for l in adj_list]
                for _ in range(k - 1):
                    for i in range(n):
                        random.shuffle(adj_list[i])
                    p = bfs_path(src, dst, forbidden)
                    if p:
                        key = tuple(p)
                        if key not in seen:
                            seen.add(key)
                            cands.append(p)
                    if len(cands) >= k:
                        break
                for i in range(n):
                    adj_list[i] = saved[i]
            return cands

        # DFS
        state = []  # list of (pair, path)
        used = set()
        nodes_visited = [0]
        result = [None]

        def dfs(idx):
            if result[0] is not None:
                return True
            if time_up():
                return False
            if nodes_visited[0] > max_nodes:
                return False
            if idx == m:
                result[0] = {p: path for (p, path) in state}
                return True
            u, v = pairs[idx]
            cands = gen_paths(u, v, used, k=4)
            for path in cands:
                nodes_visited[0] += 1
                interior = set(path[1:-1])
                if interior & bset or interior & used:
                    continue
                state.append(((u, v), path))
                used |= interior
                if dfs(idx + 1):
                    return True
                state.pop()
                used -= interior
            return False

        if dfs(0):
            return result[0]
        return None

    def validate_and_build(branches, paths):
        bset = set(branches)
        used = set()
        for (u, v), p in paths.items():
            if p[0] != u or p[-1] != v:
                return None
            interior = set(p[1:-1])
            if interior & bset:
                return None
            if interior & used:
                return None
            used |= interior
            for a, b in zip(p, p[1:]):
                if b not in adjset[a]:
                    return None
        path_list = []
        for u, v in combinations(branches, 2):
            path_list.append(paths[(u, v)])
        return {'branches': list(branches), 'paths': path_list}

    # Find a good starting clique (max clique) to seed branch sets
    def max_clique_heuristic():
        # greedy: sort by degree, add if connected to all current
        order = sorted(range(n), key=lambda i: -deg[i])
        clique = []
        for v in order:
            if all(v in adjset[u] for u in clique):
                clique.append(v)
        return clique

    seed_clique = max_clique_heuristic()
    # Ensure at least a triangle; if seed too small, expand via local search later
    if len(seed_clique) < 3:
        # find any triangle
        found = False
        for a, b, c in combinations(range(n), 3):
            if b in adjset[a] and c in adjset[a] and c in adjset[b]:
                seed_clique = [a, b, c]
                found = True
                break
        if not found:
            u, v = P['edges'][0]
            seed_clique = [u, v]

    best = None
    best_size = len(seed_clique)

    # Greedy expansion from seed
    def greedy_expand(start):
        branches = list(start)
        # verify start is routable
        paths = try_route_with_retries(branches, n_retries=10)
        if paths is None:
            # try backtracking
            paths = try_route_bt(branches)
        if paths is None:
            return None
        result = validate_and_build(branches, paths)
        if result is None:
            return None
        # try to add vertices
        candidates = sorted([v for v in range(n) if v not in set(branches)], key=lambda v: -deg[v])
        for v in candidates:
            if time_up():
                break
            new_branches = branches + [v]
            paths2 = try_route_with_retries(new_branches, n_retries=10)
            if paths2 is None:
                paths2 = try_route_bt(new_branches, max_nodes=50000)
            if paths2 is not None:
                res2 = validate_and_build(new_branches, paths2)
                if res2 is not None:
                    branches = new_branches
                    result = res2
        return result

    r = greedy_expand(seed_clique)
    if r is not None:
        best = r
        best_size = len(r['branches'])

    # Randomized search: try random branch sets of various sizes, and
    # local search (swap) on the best found.
    rng = random.Random(12345)

    def score_set(branches):
        paths = try_route_with_retries(branches, n_retries=15)
        if paths is None:
            paths = try_route_bt(branches, max_nodes=100000)
        if paths is None:
            return None
        return validate_and_build(branches, paths)

    # Try to improve by local search: swap a branch with a non-branch
    if best is not None:
        for _ in range(200):
            if time_up():
                break
            cur = best['branches']
            cur_set = set(cur)
            # pick a random branch to drop and a random non-branch to add
            b = rng.choice(cur)
            non = [v for v in range(n) if v not in cur_set]
            if not non:
                break
            v = rng.choice(non)
            new = [x for x in cur if x != b] + [v]
            r2 = score_set(new)
            if r2 is not None and len(r2['branches']) > best_size:
                best = r2
                best_size = len(r2['branches'])
            elif r2 is not None:
                # same size, but maybe enables further growth; keep as candidate
                pass

    # Randomized branch-set search for larger sizes
    candidates = sorted(range(n), key=lambda i: (-deg[i], i))
    for size in range(min(n, 14), max(best_size, 2), -1):
        if time_up():
            break
        if size <= best_size:
            break
        # try a limited number of random subsets of this size from top candidates
        top = candidates[:min(n, 18)]
        tried = 0
        max_tries = 300
        while tried < max_tries and not time_up():
            tried += 1
            combo = sorted(rng.sample(top, size))
            r2 = score_set(combo)
            if r2 is not None:
                best = r2
                best_size = size
                break

    # If still small, do a broader random search
    if best_size < 5 and not time_up():
        for _ in range(2000):
            if time_up():
                break
            size = rng.randint(3, 8)
            combo = sorted(rng.sample(range(n), size))
            r2 = score_set(combo)
            if r2 is not None and len(r2['branches']) > best_size:
                best = r2
                best_size = len(r2['branches'])

    # Fallback
    if best is None:
        # find any triangle
        for a, b, c in combinations(range(n), 3):
            if b in adjset[a] and c in adjset[a] and c in adjset[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            u, v = P['edges'][0]
            best = {'branches': [u, v], 'paths': [[u, v]]}

    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
