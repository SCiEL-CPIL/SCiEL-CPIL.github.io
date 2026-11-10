import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 83  # safety margin before 90s kill

    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    degree = [len(adj_set[i]) for i in range(n)]

    def time_check():
        return time.time() > deadline

    def k_core(k):
        """Return set of vertices in the k-core (each has >= k neighbors in core)."""
        if k <= 0:
            return set(range(n))
        core = set(range(n))
        deg = {v: len([x for x in adj_list[v] if x in core]) for v in core}
        changed = True
        while changed:
            changed = False
            for v in list(core):
                if deg[v] < k:
                    core.remove(v)
                    for w in adj_list[v]:
                        if w in core:
                            deg[w] -= 1
                    changed = True
        return core

    def bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None

    def match_internals(feasible):
        """Bipartite matching: assign distinct internal vertex to each non-adjacent pair.
        feasible: list of (pair, [candidates]). Returns dict pair->w or None."""
        if not feasible:
            return {}
        sorted_f = sorted(feasible, key=lambda x: len(x[1]))
        assignment = {}
        used = set()

        def bt(idx):
            if time_check():
                return False
            if idx == len(sorted_f):
                return True
            pair, cands = sorted_f[idx]
            for w in cands:
                if w not in used:
                    used.add(w)
                    assignment[pair] = w
                    if bt(idx + 1):
                        return True
                    used.discard(w)
                    del assignment[pair]
            return False

        if bt(0):
            return assignment
        return None

    def route(branches):
        """Route disjoint internal paths for all pairs. Returns dict pair->path or None."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        pair_info = {}
        non_direct = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                pair_info[(u, v)] = ('need', common)
                non_direct.append((u, v))
        for (u, v) in non_direct:
            if len(pair_info[(u, v)][1]) == 0:
                return None

        # Tier 1: cheap length-2 matching
        if non_direct:
            feasible = [((u, v), list(pair_info[(u, v)][1])) for (u, v) in non_direct]
            matching = match_internals(feasible)
            if matching is not None:
                assignment = {p: [w] for p, w in matching.items()}
            else:
                # Tier 2: BFS backtracking for longer paths
                assignment = {}
                used_internals = set()
                sorted_nd = sorted(non_direct, key=lambda p: len(pair_info[p][1]))

                def bt_paths(idx):
                    if time_check():
                        return False
                    if idx == len(sorted_nd):
                        return True
                    u, v = sorted_nd[idx]
                    blocked = branch_set | used_internals
                    blocked.discard(u)
                    blocked.discard(v)
                    path = bfs_path(u, v, blocked)
                    if path is None:
                        return False
                    internal = path[1:-1]
                    if len(internal) != len(set(internal)):
                        return False
                    for w in internal:
                        if w in branch_set:
                            return False
                    for w in internal:
                        used_internals.add(w)
                    assignment[(u, v)] = internal
                    if bt_paths(idx + 1):
                        return True
                    for w in internal:
                        used_internals.discard(w)
                    del assignment[(u, v)]
                    return False

                if not bt_paths(0):
                    return None
        else:
            assignment = {}

        result = {}
        for (u, v) in pairs:
            if pair_info[(u, v)][0] == 'direct':
                result[(u, v)] = [u, v]
            else:
                result[(u, v)] = [u] + assignment[(u, v)] + [v]
        return result

    def solve():
        best_result = None
        best_k = 0

        # Precompute dense cores for pruning
        cores = {}
        for k in range(2, 11):
            cores[k] = k_core(k - 1)  # branch of size k needs ~k-1 local density

        # Try k from high to low
        for k in range(8, 1, -1):
            if time_check():
                break
            # Candidate pool: prefer vertices in the (k-1)-core (dense), plus high-degree
            core = cores.get(k, set())
            if core:
                pool = list(core)
            else:
                pool = list(range(n))
            # Ensure enough candidates
            if len(pool) < k:
                pool = list(range(n))

            total_combos = 1
            for i in range(k):
                total_combos *= (len(pool) - i)
                total_combos //= (i + 1)

            found = False
            if total_combos <= 40000:
                for combo in combinations(pool, k):
                    if time_check():
                        break
                    paths = route(list(combo))
                    if paths is not None:
                        best_result = {'branches': list(combo),
                                       'paths': [paths[p] for p in sorted(paths.keys())]}
                        best_k = k
                        found = True
                        break
            else:
                # sample, but bias toward core vertices
                import random
                rng = random.Random(k * 131 + 7)
                for attempt in range(30000):
                    if time_check():
                        break
                    # mix: mostly core, some high-degree
                    if core and rng.random() < 0.8:
                        combo = tuple(rng.sample(core, k)) if len(core) >= k else None
                    if combo is None:
                        combo = tuple(rng.sample(range(n), k))
                    paths = route(list(combo))
                    if paths is not None:
                        best_result = {'branches': list(combo),
                                       'paths': [paths[p] for p in sorted(paths.keys())]}
                        best_k = k
                        found = True
                        break
            if found:
                break

        if best_result is not None:
            return best_result

        # Fallback: 2 adjacent branches
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [int(0), int(1)], 'paths': [[int(0), int(1)]]}

    res = solve()
    # Enforce exact types
    res['branches'] = [int(x) for x in res['branches']]
    res['paths'] = [[int(x) for x in path] for path in res['paths']]
    return res
# EVOLVE-BLOCK-END
