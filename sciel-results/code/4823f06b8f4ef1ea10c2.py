"""Longest simple path in a pinned sparse graph using exact matching + local search."""
import time
import random
import sys
from collections import defaultdict

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGE_LIST = P['edges']


def _build_adj():
    adj = [set() for _ in range(N)]
    for u, v in EDGE_LIST:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _hungarian(cost_matrix):
    """Solve the assignment problem (minimize cost). Returns (assignment, total_cost)
    where assignment[i] = j means row i is assigned to column j.
    Uses the Jonker-Volgenant / Hungarian algorithm.
    """
    n = len(cost_matrix)
    if n == 0:
        return [], 0
    
    # Convert to a list of lists for speed
    c = [list(row) for row in cost_matrix]
    
    # Hungarian algorithm (O(n^3))
    INF = float('inf')
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row assigned to column j
    way = [0] * (n + 1)
    
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
            if p[j0] == 0:
                break
        
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    
    # Build assignment: for each row i, find column j such that p[j] == i
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    
    total_cost = sum(c[i][assignment[i]] for i in range(n))
    return assignment, total_cost


def _path_from_matching(adj, assignment, n):
    """Given a permutation (assignment) of vertices, try to extract a long path.
    The assignment represents a 2-regular graph (union of cycles). We break cycles
    to form paths and then try to extend.
    """
    # assignment[i] = j means vertex i is matched to vertex j
    # This forms a set of cycles (since it's a permutation)
    # Each cycle is a closed walk; we break it at one point to get a path
    
    visited = [False] * n
    best_path = []
    
    for start in range(n):
        if visited[start]:
            continue
        # Trace the cycle starting from start
        cycle = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = assignment[cur]
        
        # cycle is a closed cycle: cycle[0] -> cycle[1] -> ... -> cycle[-1] -> cycle[0]
        # Break it to form a path. Try all break points and pick the best.
        best_break_path = []
        for break_idx in range(len(cycle)):
            # Path: cycle[break_idx], cycle[break_idx+1], ..., cycle[break_idx-1] (wrapping)
            path = []
            for k in range(len(cycle)):
                path.append(cycle[(break_idx + k) % len(cycle)])
            # This is a valid path if all consecutive pairs are edges
            valid = True
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    valid = False
                    break
            if valid:
                if len(path) > len(best_break_path):
                    best_break_path = path
        
        if len(best_break_path) > len(best_path):
            best_path = best_break_path
    
    return best_path


def _greedy_path(adj, start, rng):
    """Greedy path extension from start with random tie-breaking."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        neighbors = adj[cur] - seen
        if not neighbors:
            break
        scored = []
        for v in neighbors:
            deg = len(adj[v] - seen)
            scored.append((deg, rng.random(), v))
        scored.sort()
        if len(scored) >= 3 and rng.random() < 0.3:
            v = scored[rng.randint(0, 2)][2]
        else:
            v = scored[0][2]
        path.append(v)
        seen.add(v)
    return path


def _try_extend_path(adj, path, seen):
    """Try to extend the path from both ends."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        # Try extending right end
        cur = path[-1]
        neighbors = adj[cur] - seen
        if neighbors:
            v = min(neighbors, key=lambda x: (len(adj[x] - seen), x))
            path.append(v)
            seen.add(v)
            improved = True
        # Try extending left end
        if not improved:
            cur = path[0]
            neighbors = adj[cur] - seen
            if neighbors:
                v = min(neighbors, key=lambda x: (len(adj[x] - seen), x))
                path.insert(0, v)
                seen.add(v)
                improved = True
    return path


def _two_opt_improve(adj, path):
    """Attempt 2-opt style improvements: try reversing segments to expose new endpoints."""
    path = list(path)
    n = len(path)
    if n < 3:
        return path
    best = path
    best_len = n
    
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 1):
            if i < len(path) - 1:
                if path[-1] in adj[path[i]]:
                    if path[i + 1] in adj[path[i - 1]]:
                        new_path = path[:i] + path[i+1:] + [path[i]]
                        valid = True
                        for j in range(len(new_path) - 1):
                            if new_path[j+1] not in adj[new_path[j]]:
                                valid = False
                                break
                        if valid and len(new_path) == len(path):
                            new_seen = set(new_path)
                            new_path = _try_extend_path(adj, new_path, new_seen)
                            if len(new_path) > len(best):
                                best = new_path
                                best_len = len(new_path)
                                improved = True
                                break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s
    
    adj = _build_adj()
    rng = random.Random(748)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Use the matching-based approach
    # Build a cost matrix where we want to maximize the number of edges in the permutation
    # that correspond to actual graph edges.
    # cost[i][j] = 0 if (i,j) is an edge, else a large penalty
    # We minimize total cost, which maximizes the number of graph edges in the permutation.
    
    # Actually, the standard approach: create a bipartite graph with left nodes L_i and right nodes R_j.
    # Edge L_i - R_j exists if i and j are adjacent in the graph, with weight 1.
    # Also add self-loops L_i - R_i with weight 0 (to allow fixed points).
    # Find maximum weight perfect matching. The matched edges (excluding self-loops) form a 2-regular subgraph.
    
    # Let's use the Hungarian algorithm on a cost matrix.
    # cost[i][j] = 0 if i==j or (i,j) is an edge, else large value.
    # Minimizing this maximizes the number of "good" assignments.
    
    LARGE = 10000
    cost_matrix = [[LARGE] * N for _ in range(N)]
    for i in range(N):
        cost_matrix[i][i] = 0  # Self-loop always allowed, cost 0
        for j in adj[i]:
            cost_matrix[i][j] = 0  # Edge exists, cost 0
    
    # Solve the assignment problem
    assignment, total_cost = _hungarian(cost_matrix)
    
    # Extract path from the matching
    path_from_match = _path_from_matching(adj, assignment, N)
    if len(path_from_match) > best_len:
        best_len = len(path_from_match)
        best_path = path_from_match
    
    # Try to extend the path from matching
    if len(best_path) > 0:
        seen = set(best_path)
        extended = _try_extend_path(adj, best_path, seen)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Phase 2: Diverse greedy constructions from all starts
    for start in range(N):
        if time.time() > deadline:
            break
        for strategy in range(3):
            if time.time() > deadline:
                break
            if strategy == 0:
                path = _greedy_path(adj, start, rng)
            elif strategy == 1:
                # Greedy from both ends
                path = [start]
                seen = {start}
                left_ok = True
                right_ok = True
                while left_ok or right_ok:
                    if left_ok and (not right_ok or rng.random() < 0.5):
                        cur = path[0]
                        neighbors = adj[cur] - seen
                        if neighbors:
                            scored = sorted([(len(adj[v] - seen), rng.random(), v) for v in neighbors])
                            v = scored[0][2]
                            path.insert(0, v)
                            seen.add(v)
                        else:
                            left_ok = False
                    if right_ok:
                        cur = path[-1]
                        neighbors = adj[cur] - seen
                        if neighbors:
                            scored = sorted([(len(adj[v] - seen), rng.random(), v) for v in neighbors])
                            v = scored[0][2]
                            path.append(v)
                            seen.add(v)
                        else:
                            right_ok = False
            else:
                # Randomized walk
                path = [start]
                seen = {start}
                while True:
                    cur = path[-1]
                    neighbors = adj[cur] - seen
                    if not neighbors:
                        break
                    neighbor_list = list(neighbors)
                    weights = [1.0 / (len(adj[v] - seen) + 1) for v in neighbor_list]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    chosen = neighbor_list[0]
                    for v, w in zip(neighbor_list, weights):
                        cum += w
                        if r <= cum:
                            chosen = v
                            break
                    path.append(chosen)
                    seen.add(chosen)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    # Phase 3: Local improvement on best paths
    if time.time() < deadline:
        seen = set(best_path)
        improved = _try_extend_path(adj, best_path, seen)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
        
        # 2-opt improvement
        if time.time() < deadline - 5.0:
            improved = _two_opt_improve(adj, best_path)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = improved
            # Extend after 2-opt
            seen = set(best_path)
            extended = _try_extend_path(adj, best_path, seen)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended
    
    # Phase 4: More diverse searches
    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            start = rng.randint(0, N - 1)
            path = _greedy_path(adj, start, rng)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
            # Try to extend
            seen = set(path)
            extended = _try_extend_path(adj, path, seen)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended
    
    # Phase 5: Final extension attempts
    if time.time() < deadline:
        seen = set(best_path)
        final = _try_extend_path(adj, best_path, seen)
        if len(final) > best_len:
            best_len = len(final)
            best_path = final
    
    # Ensure we return a valid non-empty path
    if not best_path:
        best_path = [0]
    
    # Final validation
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            valid_len = i + 1
            best_path = best_path[:valid_len]
            break
    
    return best_path
