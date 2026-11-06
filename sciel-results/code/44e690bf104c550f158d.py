"""Solve minimum bottleneck Hamilton cycle via greedy + local search + binary search with backtracking."""
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
TOTAL_EDGES = N * (N - 1) // 2

edge_idx = [[0] * N for _ in range(N)]
for i, (u, v) in enumerate(P['edge_order'], 1):
    edge_idx[u][v] = i
    edge_idx[v][u] = i


def cycle_max_edge(cyc):
    max_e = 0
    n = len(cyc)
    for i in range(n - 1):
        e = edge_idx[cyc[i]][cyc[i+1]]
        if e > max_e:
            max_e = e
    e = edge_idx[cyc[n-1]][cyc[0]]
    if e > max_e:
        max_e = e
    return max_e


def greedy_2opt():
    path = [0]
    used = [False] * N
    used[0] = True
    for _ in range(N - 1):
        u = path[-1]
        best_v = -1
        best_idx = float('inf')
        for v in range(N):
            if not used[v]:
                idx = edge_idx[u][v]
                if idx < best_idx:
                    best_idx = idx
                    best_v = v
        path.append(best_v)
        used[best_v] = True
    current = path
    current_max = cycle_max_edge(current)
    improved = True
    while improved:
        improved = False
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                old1 = edge_idx[current[i-1]][current[i]]
                old2 = edge_idx[current[j-1]][current[j]]
                new1 = edge_idx[current[i-1]][current[j]]
                new2 = edge_idx[current[j-1]][current[i]]
                if max(new1, new2) < current_max:
                    current[i:j] = reversed(current[i:j])
                    current_max = cycle_max_edge(current)
                    improved = True
    return current


def backtracking_hamilton(threshold, deadline):
    """Try to find a Hamilton cycle using only edges with arrival index <= threshold."""
    adj = [set() for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if u != v and edge_idx[u][v] <= threshold:
                adj[u].add(v)
    for u in range(N):
        if len(adj[u]) < 2:
            return None
    # Connectivity check
    visited = [False] * N
    stack = [0]
    visited[0] = True
    cnt = 0
    while stack:
        u = stack.pop()
        cnt += 1
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    if cnt < N:
        return None
    # Backtracking with MRV
    path = [0]
    used = [False] * N
    used[0] = True
    def bt(pos):
        if time.time() > deadline:
            return False
        if pos == N - 1:
            return 0 in adj[path[-1]]
        u = path[-1]
        neighbors = [v for v in adj[u] if not used[v]]
        neighbors.sort(key=lambda v: len([w for w in adj[v] if not used[w] and w != 0]))
        for v in neighbors:
            if pos < N - 2:
                # v must have at least one unused neighbor (not 0) to continue
                has_cont = False
                for w in adj[v]:
                    if not used[w] and w != 0:
                        has_cont = True
                        break
                if not has_cont:
                    continue
            path.append(v)
            used[v] = True
            if bt(pos + 1):
                return True
            path.pop()
            used[v] = False
        return False
    if bt(1):
        return path
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 78
    
    # Phase 1: Greedy + 2-opt
    cycle = greedy_2opt()
    best_threshold = cycle_max_edge(cycle)
    best_cycle = cycle[:]
    
    # Phase 2: Binary search for lower threshold using backtracking
    lo, hi = 1, best_threshold - 1
    while lo <= hi and time.time() < deadline - 3:
        mid = (lo + hi) // 2
        found = backtracking_hamilton(mid, deadline - 2)
        if found is not None:
            best_cycle = found
            best_threshold = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Phase 3: Try to improve with more 2-opt iterations on the best cycle
    if time.time() < deadline - 1:
        current = best_cycle[:]
        current_max = cycle_max_edge(current)
        for _ in range(5):  # Multiple passes
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for i in range(1, N - 1):
                    for j in range(i + 1, N):
                        old1 = edge_idx[current[i-1]][current[i]]
                        old2 = edge_idx[current[j-1]][current[j]]
                        new1 = edge_idx[current[i-1]][current[j]]
                        new2 = edge_idx[current[j-1]][current[i]]
                        if max(new1, new2) < current_max:
                            current[i:j] = reversed(current[i:j])
                            current_max = cycle_max_edge(current)
                            improved = True
            if current_max < best_threshold:
                best_cycle = current
                best_threshold = current_max
    
    result = best_cycle
    if sorted(result) != list(range(N)):
        return list(range(N))
    return result
