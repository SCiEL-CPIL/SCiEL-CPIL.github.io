"""Combined: clique-subdivision search + backtracking path finder + candidate-set B&B."""
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0  # safety margin under the 90s limit

    n = P['n']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v) if u < v else (v, u))

    def find_disjoint_path(start, end, blocked):
        """BFS for a simple path start->end avoiding blocked (excludes start,end)."""
        if start == end:
            return [start]
        if start in blocked or end in blocked:
            return None
        q = deque()
        q.append((start, [start]))
        visited = {start}
        while q:
            if time.time() > DEADLINE:
                return None
            cur, path = q.popleft()
            if cur == end:
                return path
            for nxt in sorted(adj[cur]):
                if nxt in blocked:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
        return None

    def find_disjoint_paths(branches, used_interior, deadline):
        """Backtracking: find internally-disjoint paths for all pairs of branches.
        Returns dict {(a,b): path} or None. Branches are endpoints; interiors
        must be disjoint and avoid all branches."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        paths = {}

        def backtrack(idx, used_int):
            if time.time() > deadline:
                return False
            if idx == len(pairs):
                return True
            u, v = pairs[idx]
            # BFS shortest path u->v avoiding used_int and other branches
            q = deque([(u, [u])])
            visited = {u}
            found = None
            while q:
                if time.time() > deadline:
                    return False
                node, path = q.popleft()
                if node == v:
                    interior = set(path[1:-1])
                    if not (interior & used_int) and not (interior & branch_set):
                        found = path
                        break
                    continue
                for nbr in sorted(adj[node]):
                    if nbr in visited:
                        continue
                    if nbr in branch_set and nbr != v:
                        continue
                    if nbr in used_int:
                        continue
                    visited.add(nbr)
                    q.append((nbr, path + [nbr]))
            if found is None:
                return False
            interior = set(found[1:-1])
            paths[(u, v)] = found
            if backtrack(idx + 1, used_int | interior):
                return True
            del paths[(u, v)]
            return False

        if backtrack(0, used_interior):
            return paths
        return None

    def paths_for_clique(combo):
        """A clique: use direct edges as paths (length 2, no interior)."""
        paths = {}
        for (a, b) in combinations(combo, 2):
            if b not in adj[a]:
                return None
            paths[(a, b)] = [a, b]
        return paths

    best_S = None
    best_paths = None

    # ---- Phase 1: find the largest clique (trivially valid subdivision) ----
    # Enumerate by descending size using combinations, but cap work.
    # For a 24-vertex dense graph, large cliques are common.
    # Use Bron–Kerbosch to enumerate maximal cliques efficiently.
    def bron_kerbosch(R, Pset, X, cliques, deadline):
        if time.time() > deadline:
            return
        if not Pset and not X:
            cliques.append(frozenset(R))
            return
        # pivot: choose vertex in Pset ∪ X with max |Pset ∩ N(v)|
        union = Pset | X
        if union:
            pivot = max(union, key=lambda v: len(Pset & adj[v]))
            candidates = [v for v in Pset if v not in adj[pivot]]
        else:
            candidates = list(Pset)
        for v in candidates:
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v], cliques, deadline)
            Pset = Pset - {v}
            X = X | {v}
            if time.time() > deadline:
                return

    cliques = []
    bron_kerbosch(frozenset(), set(range(n)), frozenset(), cliques, DEADLINE)

    # Sort cliques by size descending; try largest first.
    cliques.sort(key=lambda c: -len(c))
    for c in cliques:
        if time.time() > DEADLINE:
            break
        combo = sorted(c)
        if len(combo) < 2:
            continue
        paths = paths_for_clique(combo)
        if paths is not None:
            if best_S is None or len(combo) > len(best_S):
                best_S = combo
                best_paths = paths
            # A clique of this size is found; smaller cliques can't beat it.
            # But we also want to know if a non-clique set is larger; handle in Phase 2.
            break

    # ---- Phase 2: candidate-set branch-and-bound for possibly larger sets ----
    # (non-clique branch sets) using backtracking feasibility.
    degs = [len(adj[i]) for i in range(n)]
    cands = sorted(range(n), key=lambda i: -degs[i])

    current_best_k = len(best_S) if best_S else 0

    # Iterative deepening: try to find a feasible set of size k > current_best_k.
    # Use backtracking path finder as the feasibility oracle.
    def exists_of_size(k):
        if k < 2 or k > len(cands):
            return None
        import sys
        sys.setrecursionlimit(100000)
        best_found = None

        def dfs(start_idx, chosen):
            nonlocal best_found
            if time.time() > DEADLINE:
                return False
            if len(chosen) == k:
                paths = find_disjoint_paths(list(chosen), set(), DEADLINE)
                if paths is not None:
                    best_found = (list(chosen), paths)
                    return True
                return False
            need = k - len(chosen)
            remaining = len(cands) - start_idx
            if remaining < need:
                return False
            for i in range(start_idx, len(cands) - need + 1):
                if time.time() > DEADLINE:
                    return False
                chosen.append(cands[i])
                if dfs(i + 1, chosen):
                    chosen.pop()
                    return True
                chosen.pop()
            return False

        dfs(0, [])
        return best_found

    # Try sizes from len(cands) down to current_best_k+1.
    for k in range(len(cands), max(current_best_k, 1) + 1, -1):
        if time.time() > DEADLINE:
            break
        if k <= current_best_k:
            break
        res = exists_of_size(k)
        if res is not None:
            S, paths = res
            best_S = S
            best_paths = paths
            current_best_k = k
            break  # first success at this (largest) k is the max found

    # ---- Fallback ----
    if best_S is None or len(best_S) < 2:
        # find any triangle
        found = False
        for row in combinations(range(n), 3):
            if all(b in adj[a] for a, b in combinations(row, 2)):
                best_S = list(row)
                best_paths = {}
                for (a, b) in combinations(row, 2):
                    best_paths[(a, b)] = [a, b]
                found = True
                break
        if not found:
            u, v = P['edges'][0]
            best_S = [u, v]
            best_paths = {(u, v): [u, v]}

    # ---- Build output ----
    branches = [int(x) for x in best_S]
    if len(branches) < 2:
        u, v = P['edges'][0]
        branches = [int(u), int(v)]
        best_paths = {(u, v): [int(u), int(v)]}

    # Ensure all pairs present and valid.
    final_paths = {}
    for (a, b) in combinations(branches, 2):
        key = (a, b) if a < b else (b, a)
        p = best_paths.get(key)
        if p is None:
            # try to find a path
            blocked = set(branches) - {a, b}
            p = find_disjoint_path(a, b, blocked)
            if p is None:
                if (a, b) in edge_set or (b, a) in edge_set:
                    p = [a, b]
                else:
                    # invalid; fall back to a 2-branch edge
                    u, v = P['edges'][0]
                    branches = [int(u), int(v)]
                    final_paths = {(u, v): [int(u), int(v)]}
                    break
        final_paths[key] = p
    else:
        final_paths = best_paths

    # If we fell back, recompute branches
    if len(branches) < 2:
        u, v = P['edges'][0]
        branches = [int(u), int(v)]
        final_paths = {(u, v): [int(u), int(v)]}

    out_paths = []
    for a in branches:
        for b in branches:
            if a >= b:
                continue
            key = (a, b)
            p = final_paths.get(key)
            if p is None:
                continue
            out_paths.append([int(x) for x in p])

    return {'branches': branches, 'paths': out_paths}
