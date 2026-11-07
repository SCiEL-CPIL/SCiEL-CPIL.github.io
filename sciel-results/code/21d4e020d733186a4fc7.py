"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import deque
from itertools import combinations

sys.setrecursionlimit(100000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Precompute all-pairs shortest paths and predecessors
    INF = float('inf')
    dist = [[INF]*n for _ in range(n)]
    pred = [[-1]*n for _ in range(n)]
    for start in range(n):
        dist[start][start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[start][v] == INF:
                    dist[start][v] = dist[start][u] + 1
                    pred[start][v] = u
                    q.append(v)

    def get_shortest_path(start, end):
        if dist[start][end] == INF:
            return None
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = pred[start][cur]
        path.reverse()
        return path

    start_time = time.time()
    deadline = start_time + 50.0  # safety margin

    # For a given set B of branches, check if there exist internally disjoint paths
    # for all non-adjacent pairs. We use a backtracking search with good ordering.
    def check_clique_subdivision(B, time_limit=3.0):
        """
        Returns (success, paths_dict) where paths_dict maps (u,v) -> path for non-adjacent pairs.
        """
        B_set = set(B)
        k = len(B)
        non_adj_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj_pairs.append((u, v))

        if not non_adj_pairs:
            return True, {}

        # For each non-adjacent pair, enumerate candidate paths (shortest first)
        # We'll use BFS to find paths up to a certain length, preferring shorter ones
        domains = {}
        for (u, v) in non_adj_pairs:
            paths = []
            # BFS from u to v, avoiding B_set in interior
            visited = {u}
            queue = deque([(u, [u])])
            found = False
            while queue and not found:
                curr, path = queue.popleft()
                if len(path) > 10:  # limit path length
                    continue
                for nxt in adj_list[curr]:
                    if nxt == v:
                        paths.append(path + [v])
                        found = True
                        break
                    if nxt not in visited and nxt not in B_set:
                        visited.add(nxt)
                        queue.append((nxt, path + [nxt]))
            
            if not paths:
                return False, None
            # Sort by length
            paths.sort(key=lambda p: (len(p), p))
            domains[(u, v)] = paths[:20]  # keep top 20 candidates

        # Order pairs by domain size (MRV heuristic)
        sorted_pairs = sorted(non_adj_pairs, key=lambda p: len(domains[p]))
        used_interiors = set()
        assignment = {}

        def backtrack(idx):
            if time.time() - start_time > deadline - 5:
                return False
            if idx == len(sorted_pairs):
                return True
            u, v = sorted_pairs[idx]
            for path in domains[(u, v)]:
                interior = frozenset(path[1:-1])
                if interior & used_interiors:
                    continue
                used_interiors.update(interior)
                assignment[(u, v)] = path
                if backtrack(idx + 1):
                    return True
                used_interiors.difference_update(interior)
                del assignment[(u, v)]
            return False

        if backtrack(0):
            return True, dict(assignment)
        return False, None

    # Strategy: Start from the maximum clique, then greedily add vertices
    # that can be accommodated.

    # Find maximum clique using Bron-Kerbosch
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(n)), set())
    max_clique = list(best_clique)

    # Check if the max clique itself is a valid clique subdivision
    best_B = []
    best_paths = {}
    ok, paths_dict = check_clique_subdivision(max_clique, time_limit=5.0)
    if ok:
        best_B = max_clique
        best_paths = paths_dict

    # Greedily try to add vertices to the current best set
    if best_B:
        current_set = set(best_B)
        remaining = [v for v in range(n) if v not in current_set]
        # Sort by degree to current set (descending), then by overall degree
        remaining.sort(key=lambda v: (-len(adj[v] & current_set), -len(adj[v])))

        for v in remaining:
            if time.time() > deadline - 10:
                break
            new_set = current_set | {v}
            new_B = list(new_set)
            ok, paths_dict = check_clique_subdivision(new_B, time_limit=2.0)
            if ok:
                best_B = new_B
                best_paths = paths_dict
                current_set = new_set
                remaining.remove(v)
                remaining.sort(key=lambda x: (-len(adj[x] & current_set), -len(adj[x])))

    # Try random restarts for potentially larger solutions
    for restart in range(50):
        if time.time() > deadline - 10:
            break

        # Randomly pick a subset of size larger than current best
        target_k = min(n, len(best_B) + 1 + random.randint(0, 3))
        if target_k > n:
            continue
        B = list(random.sample(range(n), target_k))
        B.sort()

        # Quick feasibility check: all pairs must be connected
        feasible = True
        for i in range(target_k):
            for j in range(i+1, target_k):
                if dist[B[i]][B[j]] == INF:
                    feasible = False
                    break
            if not feasible:
                break
        if not feasible:
            continue

        ok, paths_dict = check_clique_subdivision(B, time_limit=1.5)
        if ok and len(B) > len(best_B):
            best_B = B
            best_paths = paths_dict

    # Try systematic search from large k downward
    if time.time() < deadline - 15:
        for k in range(n, len(best_B), -1):
            if time.time() > deadline - 10:
                break
            found = False
            # Sample some combinations
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                B = list(random.sample(range(n), k))
                B.sort()
                feasible = True
                for i in range(k):
                    for j in range(i+1, k):
                        if dist[B[i]][B[j]] == INF:
                            feasible = False
                            break
                    if not feasible:
                        break
                if not feasible:
                    continue
                ok, paths_dict = check_clique_subdivision(B, time_limit=1.0)
                if ok:
                    best_B = B
                    best_paths = paths_dict
                    found = True
                    break
            if found:
                break

    # Fallback
    if not best_B:
        best_B = [0, 1]
        best_paths = {}

    # Construct output
    B = best_B
    paths_out = []
    B_set = set(B)

    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths_out.append([int(u), int(v)])
            else:
                if (u, v) in best_paths:
                    p = best_paths[(u, v)]
                    paths_out.append([int(x) for x in p])
                elif (v, u) in best_paths:
                    p = best_paths[(v, u)]
                    paths_out.append([int(x) for x in reversed(p)])
                else:
                    sp = get_shortest_path(u, v)
                    if sp:
                        paths_out.append([int(x) for x in sp])
                    else:
                        paths_out.append([int(u), int(v)])

    return {'branches': [int(b) for b in B], 'paths': paths_out}

# EVOLVE-BLOCK-END
