import time
import sys
import random
from itertools import combinations
from collections import deque
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s kill)

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute all-pairs shortest paths (for path reconstruction and feasibility hints)
    # Also compute all simple paths up to length 4 for each pair
    def get_shortest_path(u, v, forbidden=None):
        """BFS shortest path from u to v avoiding forbidden vertices (except u,v)."""
        if forbidden is None:
            forbidden = set()
        if u == v:
            return [u]
        visited = {u}
        parent = {u: None}
        queue = deque([u])
        while queue:
            cur = queue.popleft()
            if cur == v:
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in sorted(adj[cur]):
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def flow_feasible(branches):
        """Check if vertex-disjoint interior paths exist for all pairs in branches.
        Uses max-flow on split graph as a necessary condition."""
        if len(branches) < 2:
            return True
        b_set = set(branches)
        k = len(branches)
        required_flow = k * (k - 1) // 2

        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2

        flow_adj = [[] for _ in range(total_nodes)]

        def add_flow_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

        for v in range(n):
            if v in b_set:
                add_flow_edge(2 * v, 2 * v + 1, 10**9)
            else:
                add_flow_edge(2 * v, 2 * v + 1, 1)

        for u in range(n):
            for v in adj[u]:
                if u < v:
                    add_flow_edge(2 * u + 1, 2 * v, 10**9)
                    add_flow_edge(2 * v + 1, 2 * u, 10**9)

        for b in branches:
            add_flow_edge(S, 2 * b, 10**9)
            add_flow_edge(2 * b + 1, T, 10**9)

        def bfs_max_flow():
            total_flow = 0
            while True:
                parent = [-1] * total_nodes
                parent_edge = [None] * total_nodes
                queue = deque([S])
                parent[S] = S
                found = False
                while queue and not found:
                    cur = queue.popleft()
                    for i, (to, cap, rev) in enumerate(flow_adj[cur]):
                        if cap > 0 and parent[to] == -1:
                            parent[to] = cur
                            parent_edge[to] = (i, rev)
                            if to == T:
                                found = True
                                break
                            queue.append(to)
                if not found:
                    break
                bottleneck = 10**18
                node = T
                while node != S:
                    i, rev = parent_edge[node]
                    bottleneck = min(bottleneck, flow_adj[parent[node]][i][1])
                    node = parent[node]
                node = T
                while node != S:
                    i, rev = parent_edge[node]
                    flow_adj[parent[node]][i][1] -= bottleneck
                    flow_adj[node][rev][1] += bottleneck
                    node = parent[node]
                total_flow += bottleneck
            return total_flow

        maxflow = bfs_max_flow()
        return maxflow >= required_flow

    def find_paths_greedy(branches, order=None):
        """Find paths for all pairs using greedy BFS. Returns dict or None."""
        b_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        if order is None:
            # Sort by path length (direct edges first)
            pairs.sort(key=lambda uv: (0 if uv[1] in adj[uv[0]] else 1))
        else:
            pairs = order
        paths = {}
        used_interior = set()

        for u, v in pairs:
            if v in adj[u]:
                paths[(u, v)] = [u, v]
                continue
            forbidden = b_set - {u, v}
            forbidden |= used_interior
            path = get_shortest_path(u, v, forbidden)
            if path is None:
                return None
            paths[(u, v)] = path
            for mid in path[1:-1]:
                used_interior.add(mid)
        return paths

    def find_paths_backtracking(branches, max_nodes=20000):
        """Backtracking search for vertex-disjoint paths. Returns dict or None."""
        b_set = set(branches)
        sorted_b = sorted(branches)
        pairs = list(combinations(sorted_b, 2))
        # Sort pairs: direct edges first, then by BFS distance
        def bfs_dist(u, v):
            if v in adj[u]:
                return 0
            visited = {u}
            queue = deque([(u, 0)])
            while queue:
                cur, d = queue.popleft()
                if cur == v:
                    return d
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, d + 1))
            return 999
        pairs.sort(key=lambda uv: bfs_dist(uv[0], uv[1]))

        paths = {}
        used_interior = set()
        node_count = [0]

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(pairs):
                return True
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            u, v = pairs[idx]
            forbidden = b_set - {u, v}
            forbidden |= used_interior

            if v in adj[u]:
                paths[(u, v)] = [u, v]
                if backtrack(idx + 1):
                    return True
                del paths[(u, v)]
                return False

            # Find all simple paths from u to v up to length 5 avoiding forbidden
            # Use DFS with depth limit
            found_paths = []
            def dfs_paths(cur, target, path_so_far, visited_set, depth):
                if depth > 5:
                    return
                if cur == target:
                    found_paths.append(list(path_so_far))
                    return
                for nb in sorted(adj[cur]):
                    if nb not in visited_set and nb not in forbidden:
                        visited_set.add(nb)
                        path_so_far.append(nb)
                        dfs_paths(nb, target, path_so_far, visited_set, depth + 1)
                        path_so_far.pop()
                        visited_set.remove(nb)

            dfs_paths(u, v, [u], {u}, 1)

            # Try each candidate path
            for cand in found_paths:
                interiors = set(cand[1:-1])
                if interiors & used_interior:
                    continue
                paths[(u, v)] = cand
                used_interior.update(interiors)
                if backtrack(idx + 1):
                    return True
                used_interior.difference_update(interiors)
                del paths[(u, v)]
            return False

        if backtrack(0):
            return paths
        return None

    # ---------- Phase 1: Find maximum clique (Bron-Kerbosch with pivot) ----------
    best_clique = []

    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return

    bronk(set(), set(range(n)), set())

    # ---------- Phase 2: Try to build a clique subdivision larger than the clique ----------
    # Strategy: Start from the max clique, try to add vertices one by one.
    # For each candidate addition, check feasibility via flow, then find paths
    # using backtracking.

    current_branches = list(best_clique)
    current_paths = find_paths_greedy(current_branches)
    if current_paths is None:
        # Try backtracking for the clique
        current_paths = find_paths_backtracking(current_branches)
    if current_paths is None:
        # Fallback: direct edges (only valid if clique)
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]

    best_branches = list(current_branches)
    best_paths = dict(current_paths)

    # Try to add vertices one at a time
    remaining = [v for v in range(n) if v not in set(current_branches)]
    # Sort remaining by degree (higher degree first - more likely to connect)
    remaining.sort(key=lambda v: -len(adj[v]))

    improved = True
    while improved:
        improved = False
        for v in remaining[:]:
            if time.time() > deadline - 5:
                break
            test_branches = current_branches + [v]
            if not flow_feasible(test_branches):
                continue
            # Try to find paths
            test_paths = find_paths_greedy(test_branches)
            if test_paths is None:
                test_paths = find_paths_backtracking(test_branches)
            if test_paths is not None:
                current_branches = test_branches
                current_paths = test_paths
                remaining.remove(v)
                improved = True
                best_branches = list(current_branches)
                best_paths = dict(current_paths)

    # ---------- Phase 3: Local search with swaps ----------
    if time.time() < deadline - 15:
        for round_num in range(5):
            if time.time() > deadline - 5:
                break
            swap_improved = False
            for i in range(len(current_branches)):
                if time.time() > deadline - 5:
                    break
                for v in remaining:
                    if time.time() > deadline - 5:
                        break
                    test_branches = current_branches[:i] + current_branches[i+1:] + [v]
                    if not flow_feasible(test_branches):
                        continue
                    test_paths = find_paths_greedy(test_branches)
                    if test_paths is None:
                        test_paths = find_paths_backtracking(test_branches)
                    if test_paths is not None:
                        removed = current_branches[i]
                        current_branches = test_branches
                        current_paths = test_paths
                        remaining.remove(v)
                        remaining.append(removed)
                        swap_improved = True
                        best_branches = list(current_branches)
                        best_paths = dict(current_paths)
                        break
                if swap_improved:
                    break
            if not swap_improved:
                break

    # ---------- Phase 4: Try different starting points with random restarts ----------
    if time.time() < deadline - 20:
        random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Start from a random subset of size best_size - 1
            all_vertices = list(range(n))
            random.shuffle(all_vertices)
            start_set = all_vertices[:max(2, len(best_branches) - 2)]
            start_set = list(start_set)
            
            test_paths = find_paths_greedy(start_set)
            if test_paths is None:
                test_paths = find_paths_backtracking(start_set)
            if test_paths is None:
                continue

            trial_branches = list(start_set)
            trial_paths = test_paths
            trial_remaining = [v for v in range(n) if v not in set(trial_branches)]
            trial_remaining.sort(key=lambda v: -len(adj[v]))

            trial_improved = True
            while trial_improved:
                trial_improved = False
                for v in trial_remaining[:]:
                    if time.time() > deadline - 5:
                        break
                    tb = trial_branches + [v]
                    if not flow_feasible(tb):
                        continue
                    tp = find_paths_greedy(tb)
                    if tp is None:
                        tp = find_paths_backtracking(tb)
                    if tp is not None:
                        trial_branches = tb
                        trial_paths = tp
                        trial_remaining.remove(v)
                        trial_improved = True

            if len(trial_branches) > len(best_branches):
                best_branches = list(trial_branches)
                best_paths = dict(trial_paths)
                current_branches = list(trial_branches)
                current_paths = trial_paths

    # ---------- Build final result ----------
    final_branches = best_branches
    final_paths = best_paths

    if final_paths is None or len(final_branches) < 2:
        # Ultimate fallback: use the clique with direct edges
        final_branches = list(best_clique)
        final_paths = {}
        for u, v in combinations(sorted(final_branches), 2):
            final_paths[(u, v)] = [u, v]

    # Verify and build output
    branches = [int(b) for b in sorted(final_branches)]
    paths_list = []
    for u, v in combinations(sorted(final_branches), 2):
        p = final_paths.get((u, v), [u, v])
        # Verify path validity
        if p[0] != u or p[-1] != v:
            p = [u, v]  # fallback to direct edge
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}
