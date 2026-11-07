"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    start_time = time.time()
    deadline = start_time + 85.0
    random.seed(12345)

    def find_shortest_path(src, dst, forbidden):
        if src == dst:
            return [src]
        if dst in forbidden:
            return None
        visited = {src}
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
                if nb == dst:
                    parent[nb] = cur
                    return None  # handled below by checking
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_k_paths(src, dst, forbidden, k=6):
        """Find up to k simple paths from src to dst avoiding forbidden."""
        if src == dst:
            return [[src]]
        if dst in forbidden:
            return []
        results = []
        # BFS for shortest, then grow with a bounded DFS for alternatives
        # 1) shortest via BFS
        p = find_shortest_path(src, dst, forbidden)
        if p is not None:
            results.append(p)
        # 2) DFS with a small depth bound to find a few alternatives
        if len(results) < k:
            depth_limit = 8
            def dfs(node, path, used):
                if len(results) >= k:
                    return
                if node == dst:
                    results.append(list(path))
                    return
                if len(path) > depth_limit:
                    return
                for nb in adj[node]:
                    if nb in used:
                        continue
                    if nb in forbidden:
                        continue
                    used.add(nb)
                    path.append(nb)
                    dfs(nb, path, used)
                    path.pop()
                    used.discard(nb)
                    if len(results) >= k:
                        return
            used = {src}
            dfs(src, [src], used)
        # de-duplicate
        uniq = []
        seen = set()
        for p in results:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def make_candidates(branches, deadline):
        """For each unordered pair of branches, build candidate paths.
        Returns dict pair->list_of_paths, or None if some pair has no path."""
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        cand = {}
        for u, v in pairs:
            forbidden = branch_set - {u, v}
            paths = find_k_paths(u, v, forbidden, k=6)
            if not paths:
                return None
            # direct edge first if present
            paths.sort(key=lambda p: (len(p), p))
            cand[(u, v)] = paths
        return cand

    def solve_conflict(cand, branches, deadline):
        """Independent-set style: pick one path per pair, no shared interior vertices.
        Returns dict pair->path or None."""
        branch_set = set(branches)
        pairs = list(cand.keys())
        # order by fewest candidates first (most constrained)
        order = sorted(pairs, key=lambda p: len(cand[p]))

        # Precompute interior sets
        interiors = {}
        for p in pairs:
            interiors[p] = [set(pp[1:-1]) for pp in cand[p]]

        assignment = {}
        used = set()

        def rec(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True
            p = order[idx]
            for i, path in enumerate(cand[p]):
                inter = interiors[p][i]
                if inter & used:
                    continue
                if inter & branch_set:
                    continue
                assignment[p] = path
                used.update(inter)
                if rec(idx + 1):
                    return True
                used.difference_update(inter)
                del assignment[p]
            return False

        if rec(0):
            return assignment
        return None

    def greedy_conflict(cand, branches, deadline):
        """Greedy independent set: repeatedly take the pair with the most
        restrictive free path. Returns dict or None."""
        branch_set = set(branches)
        pairs = list(cand.keys())
        remaining = list(pairs)
        assignment = {}
        used = set()
        while remaining:
            if time.time() > deadline:
                return None
            # score each remaining pair by its best free path (shortest)
            best = None
            best_path = None
            best_len = None
            for p in remaining:
                for path in cand[p]:
                    inter = set(path[1:-1])
                    if inter & used or inter & branch_set:
                        continue
                    L = len(path)
                    if best_path is None or L < best_len:
                        best = p
                        best_path = path
                        best_len = L
            if best is None:
                return None
            assignment[best] = best_path
            used.update(set(best_path[1:-1]))
            remaining.remove(best)
        return assignment

    def build(branches, deadline):
        cand = make_candidates(branches, deadline)
        if cand is None:
            return None
        res = solve_conflict(cand, branches, deadline)
        if res is None:
            res = greedy_conflict(cand, branches, deadline)
        return res

    def branches_from_assignment(assignment):
        verts = set()
        for p in assignment:
            verts.add(p[0])
            verts.add(p[1])
        return sorted(verts)

    def verify(branches, paths):
        branch_set = set(branches)
        if len(branches) < 2:
            return False
        used = set()
        for path in paths:
            if len(path) < 2:
                return False
            if path[0] not in branch_set or path[-1] not in branch_set:
                return False
            if path[0] == path[-1]:
                return False
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return False
            inter = set(path[1:-1])
            if inter & branch_set:
                return False
            if inter & used:
                return False
            used.update(inter)
        return True

    best = None
    best_count = 2
    # baseline: first edge
    u0, v0 = P['edges'][0]
    best = {'branches': [int(u0), int(v0)], 'paths': [[int(u0), int(v0)]]}

    def try_update(branches, assignment):
        nonlocal best, best_count
        if assignment is None:
            return
        branches = sorted(branches)
        if len(branches) <= best_count:
            return
        paths_list = []
        for (a, b) in itertools.combinations(branches, 2):
            paths_list.append(assignment[(a, b)])
        if verify(branches, paths_list):
            best_count = len(branches)
            best = {'branches': [int(x) for x in branches],
                    'paths': [[int(x) for x in p] for p in paths_list]}

    # ---- Global greedy independent set over ALL pairs (direct optimization) ----
    def global_greedy(deadline):
        all_pairs = list(itertools.combinations(range(n), 2))
        # candidate: shortest path for each pair (with both endpoints allowed)
        pair_path = {}
        for (u, v) in all_pairs:
            p = find_shortest_path(u, v, set())
            if p is not None:
                pair_path[(u, v)] = p
        # Build conflict among interior vertices
        # Greedy: repeatedly add the path that touches fewest new interior vertices
        chosen = {}
        used = set()
        remaining = list(pair_path.keys())
        while remaining:
            if time.time() > deadline:
                break
            best_p = None
            best_cost = None
            for p in remaining:
                inter = set(pair_path[p][1:-1])
                if inter & used:
                    continue
                cost = len(inter)
                if best_p is None or cost < best_cost:
                    best_p = p
                    best_cost = cost
            if best_p is None:
                break
            chosen[best_p] = pair_path[best_p]
            used.update(set(pair_path[best_p][1:-1]))
            remaining.remove(best_p)
        # chosen is a set of pairwise-interior-disjoint paths; but we need one per
        # pair of a common branch set (a clique of endpoints). Find max branch set
        # whose all pairs are present in chosen.
        verts = set()
        for p in chosen:
            verts.add(p[0]); verts.add(p[1])
        # find largest subset S of verts such that every pair in S is in chosen
        verts = sorted(verts)
        # greedy grow
        S = []
        for v in verts:
            cand = S + [v]
            ok = True
            for a, b in itertools.combinations(cand, 2):
                key = (min(a, b), max(a, b))
                if key not in chosen:
                    ok = False
                    break
            if ok:
                S = cand
        return S, {k: v for k, v in chosen.items()}

    S, gassign = global_greedy(deadline)
    if S:
        try_update(S, gassign)

    # ---- Bron-Kerbosch to enumerate maximal cliques (branch sets must be cliques
    #     for direct-edge guarantee, but we still run path search) ----
    cliques = []

    def bronk(R, P, X, deadline):
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return
        if P or X:
            pivot = max(P | X, key=lambda v: len(adj[v] & P))
        else:
            return
        for v in list(P):
            if v in adj[pivot]:
                continue
            bronk(R | {v}, P & adj[v], X & adj[v], deadline)
            P.discard(v)
            X.add(v)

    bronk(set(), set(range(n)), set(), deadline)
    # sort cliques by size desc, try building
    cliques.sort(key=lambda c: -len(c))
    for c in cliques:
        if time.time() > deadline:
            break
        if len(c) <= best_count:
            break
        res = build(c, deadline)
        try_update(c, res)

    # ---- Local search / random sets ----
    for size in range(3, min(8, n + 1)):
        if time.time() > deadline:
            break
        for _ in range(120):
            if time.time() > deadline:
                break
            rand_set = random.sample(range(n), size)
            res = build(rand_set, deadline)
            try_update(rand_set, res)

    # ---- Grow from best ----
    if best_count >= 3:
        current = list(best['branches'])
        for _ in range(40):
            if time.time() > deadline:
                break
            progress = False
            for cand in random.sample(range(n), 12):
                if time.time() > deadline:
                    break
                if cand in current:
                    continue
                newb = current + [cand]
                res = build(newb, deadline)
                if res is not None:
                    nb = sorted(newb)
                    if verify(nb, [res[p] for p in itertools.combinations(nb, 2)]):
                        current = nb
                        progress = True
                        try_update(nb, res)
                        break
            if not progress:
                break

    # Final verify & return
    if best is not None and verify(best['branches'], best['paths']):
        return {'branches': [int(b) for b in best['branches']],
                'paths': [[int(v) for v in p] for p in best['paths']]}
    return {'branches': [int(u0), int(v0)], 'paths': [[int(u0), int(v0)]]}

# EVOLVE-BLOCK-END
