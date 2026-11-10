"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
adj_list = [sorted(adj[i]) for i in range(N)]

DEADLINE = time.time() + 78

def time_left():
    return DEADLINE - time.time()

def greedy_coloring(candidates, adj_set):
    colors = {}
    for v in candidates:
        used = set()
        for u in candidates:
            if u != v and u in colors and u in adj_set[v]:
                used.add(colors[u])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    if not colors:
        return 0, colors
    return max(colors.values()) + 1, colors

def max_clique_branch_bound(adj_set, deadline):
    n = len(adj_set)
    best_clique = []

    def expand(candidates, current):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > len(best_clique):
                best_clique = list(current)
            return
        num_colors, color_map = greedy_coloring(candidates, adj_set)
        if len(current) + num_colors <= len(best_clique):
            return
        for i in range(len(candidates) - 1, -1, -1):
            if time.time() > deadline:
                return
            v = candidates[i]
            if len(current) + i + 1 <= len(best_clique):
                break
            new_candidates = [u for u in candidates[:i] if u in adj_set[v]]
            new_current = current + [v]
            expand(new_candidates, new_current)
            candidates = candidates[:i]
            num_colors, _ = greedy_coloring(candidates, adj_set)
            if len(current) + num_colors <= len(best_clique):
                return

    expand(list(range(n)), [])
    return best_clique

def route_once(branches, branch_set, pair_order, neighbor_perm):
    """Single routing attempt with a fixed pair order and neighbor permutation.
    Returns dict {(i,j): path} or None."""
    used_interior = set()
    paths = {}
    for (i, j) in pair_order:
        s, t = branches[i], branches[j]
        parent = {}
        visited = {s}
        q = deque([s])
        found = False
        while q:
            u = q.popleft()
            if u == t:
                found = True
                break
            for w in neighbor_perm[u]:
                if w in visited:
                    continue
                if w in branch_set and w != t:
                    continue
                if w in used_interior:
                    continue
                visited.add(w)
                parent[w] = u
                q.append(w)
        if not found:
            return None
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        for v in path[1:-1]:
            used_interior.add(v)
        paths[(i, j)] = path
    result = []
    for (i, j) in combinations(range(len(branches)), 2):
        result.append(paths[(i, j)])
    return result

def route_multi(branches, deadline, rng, n_tries=60):
    """Try multiple randomized routings. Returns (path_list, branch_list) or (None, None)."""
    n_b = len(branches)
    if n_b < 2:
        return None, None
    branch_set = set(branches)
    pairs = list(combinations(range(n_b), 2))
    base_neighbor = [list(adj_list[v]) for v in range(N)]

    def key_direct(p):
        u, v = branches[p[0]], branches[p[1]]
        return 0 if v in adj[u] else 1

    # Deterministic order: direct edges first
    det_pairs = sorted(pairs, key=key_direct)
    for order in [det_pairs]:
        if time_left() < 1:
            break
        res = route_once(branches, branch_set, order, base_neighbor)
        if res is not None:
            return res, branches

    # Randomized orders
    for _ in range(n_tries):
        if time_left() < 1:
            break
        order = list(pairs)
        rng.shuffle(order)
        # bias: put direct edges earlier
        order.sort(key=lambda p: (rng.random(), key_direct(p)))
        perm = [list(x) for x in base_neighbor]
        for v in range(N):
            rng.shuffle(perm[v])
        res = route_once(branches, branch_set, order, perm)
        if res is not None:
            return res, branches
    return None, None

def try_branch_set(branches, deadline, rng):
    res, br = route_multi(branches, deadline, rng)
    return res

def construct():
    deadline = DEADLINE
    rng = random.Random(12345)
    best_branches = None
    best_paths = None

    def consider(branches):
        nonlocal best_branches, best_paths
        if branches is None:
            return
        if best_branches is not None and len(branches) <= len(best_branches):
            return
        paths = try_branch_set(branches, deadline, rng)
        if paths is not None:
            best_branches = list(branches)
            best_paths = paths

    # Step 1: maximum clique
    max_clique = max_clique_branch_bound(adj, deadline - 25)

    if len(max_clique) >= 2:
        consider(max_clique)
        # Systematic subset search within the clique, decreasing size
        mc = max_clique
        for size in range(len(mc) - 1, 1, -1):
            if time_left() < 6:
                break
            if best_branches is not None and size <= len(best_branches):
                break
            for subset in combinations(mc, size):
                if time_left() < 3:
                    break
                consider(list(subset))

    # Step 2: seeds from high-degree vertices
    deg_sorted = sorted(range(N), key=lambda v: -len(adj[v]))
    tried = set()
    for i in range(min(12, len(deg_sorted))):
        if time_left() < 6:
            break
        for j in range(i + 1, min(12, len(deg_sorted))):
            if time_left() < 3:
                break
            seed = [deg_sorted[i], deg_sorted[j]]
            key = tuple(sorted(seed))
            if key in tried:
                continue
            tried.add(key)
            consider(seed)

    # Step 3: extend the best found by adding one vertex
    if best_branches is not None and time_left() > 8:
        base = list(best_branches)
        improved = True
        while improved and time_left() > 5:
            improved = False
            for cand in range(N):
                if cand in base:
                    continue
                if time_left() < 3:
                    break
                trial = list(base) + [cand]
                if best_branches is not None and len(trial) <= len(best_branches):
                    continue
                paths = try_branch_set(trial, deadline, rng)
                if paths is not None:
                    best_branches = trial
                    best_paths = paths
                    base = trial
                    improved = True
                    break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        for u, v in P['edges']:
            if v in adj[u]:
                best_branches = [u, v]
                best_paths = [[u, v]]
                break

    path_list = [list(p) for p in best_paths] if best_paths is not None else []
    return {'branches': [int(b) for b in best_branches], 'paths': path_list}
# EVOLVE-BLOCK-END
