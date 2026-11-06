"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Precompute edge position (1-based index)
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = (min(u, v), max(u, v))
    edge_pos[key] = i + 1

def find_hamilton_cycle_k(K, deadline, start_vertex=0, use_warnsdorff=True):
    """Try to find a Hamilton cycle using only edges with position <= K."""
    # Build adjacency from first K edges
    adj = [set() for _ in range(N)]
    for i in range(K):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    
    # Check each vertex has degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Check connectivity of the graph
    visited_conn = [False] * N
    stack = [start_vertex]
    visited_conn[start_vertex] = True
    count = 1
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if not visited_conn[nb]:
                visited_conn[nb] = True
                count += 1
                stack.append(nb)
    if count < N:
        return None
    
    # Precompute sorted neighbor lists for faster iteration
    adj_sorted = [sorted(adj[v]) for v in range(N)]
    
    # Backtracking DFS to find Hamilton cycle
    path = [start_vertex]
    visited = [False] * N
    visited[start_vertex] = True
    unvisited_count = N - 1
    
    def get_candidates(last, path_len):
        """Get candidate next vertices with Warnsdorff heuristic and pruning."""
        remaining = N - path_len
        candidates = []
        for nxt in adj_sorted[last]:
            if visited[nxt]:
                continue
            if remaining == 1:
                # Last vertex must connect to start
                if start_vertex in adj[nxt]:
                    candidates.append((0, nxt))
            else:
                # Warnsdorff: count unvisited neighbors
                uc = 0
                for nb in adj_sorted[nxt]:
                    if not visited[nb]:
                        uc += 1
                if uc >= 1:
                    candidates.append((uc, nxt))
        if not candidates:
            return []
        # Sort by unvisited count (ascending) - Warnsdorff's rule
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates]
    
    def is_feasible():
        """Check if the partial path can be extended to a Hamilton cycle."""
        if time.time() > deadline:
            return False
        # Check that all unvisited vertices have at least one unvisited neighbor
        # or are the start vertex (which can be connected at the end)
        last_in_path = len(path)
        for v in range(N):
            if not visited[v] and v != start_vertex:
                has_unvisited_neighbor = False
                for nb in adj_sorted[v]:
                    if not visited[nb]:
                        has_unvisited_neighbor = True
                        break
                if not has_unvisited_neighbor:
                    # v must be connectable to start at the end
                    if last_in_path == N - 1 and start_vertex in adj[v]:
                        continue
                    return False
        return True
    
    def dfs():
        nonlocal unvisited_count
        if time.time() > deadline:
            return None
        if len(path) == N:
            last = path[-1]
            if start_vertex in adj[last]:
                return path[:]
            return None
        last = path[-1]
        candidates = get_candidates(last, len(path))
        for nxt in candidates:
            path.append(nxt)
            visited[nxt] = True
            unvisited_count -= 1
            if is_feasible():
                result = dfs()
                if result is not None:
                    return result
            path.pop()
            visited[nxt] = False
            unvisited_count += 1
        return None
    
    return dfs()

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Fallback: identity permutation
    best_cycle = list(range(N))
    
    found = False
    best_K = float('inf')
    
    # First, find any valid cycle with a large K to have a fallback
    # Try decreasing K values until we find a cycle
    for K_try in [150, 120, 100, 80, 60, 50, 40, 35, 30, 28, 26, 24]:
        if time.time() > deadline - 8:
            break
        for sv in range(N):
            if time.time() > deadline - 8:
                break
            cycle = find_hamilton_cycle_k(K_try, deadline, start_vertex=sv)
            if cycle is not None:
                # Verify
                max_pos = 0
                for i in range(N):
                    u = cycle[i]
                    v = cycle[(i + 1) % N]
                    key = (min(u, v), max(u, v))
                    pos = edge_pos.get(key, 999)
                    max_pos = max(max_pos, pos)
                if max_pos <= K_try:
                    found = True
                    best_cycle = cycle
                    best_K = max_pos
                    break
        if found:
            break
    
    if not found:
        # Fallback: return identity (should not happen with K=150)
        return best_cycle
    
    # Now try to improve: binary search for smaller K
    if time.time() < deadline - 15:
        lo = 24
        hi = best_K - 1
        while lo <= hi and time.time() < deadline - 8:
            mid = (lo + hi) // 2
            success = False
            for sv in range(N):
                if time.time() > deadline - 8:
                    break
                cycle = find_hamilton_cycle_k(mid, deadline, start_vertex=sv)
                if cycle is not None:
                    # Verify
                    max_pos = 0
                    for i in range(N):
                        u = cycle[i]
                        v = cycle[(i + 1) % N]
                        key = (min(u, v), max(u, v))
                        pos = edge_pos.get(key, 999)
                        max_pos = max(max_pos, pos)
                    if max_pos <= mid:
                        success = True
                        best_cycle = cycle
                        best_K = max_pos
                        break
            if success:
                hi = mid - 1
            else:
                lo = mid + 1
    
    # Final polish: try to find an even better cycle with the current best_K
    # by trying all start vertices and verifying the actual max position
    if time.time() < deadline - 5:
        for sv in range(N):
            if time.time() > deadline - 5:
                break
            cycle = find_hamilton_cycle_k(best_K, deadline, start_vertex=sv)
            if cycle is not None:
                max_pos = 0
                for i in range(N):
                    u = cycle[i]
                    v = cycle[(i + 1) % N]
                    key = (min(u, v), max(u, v))
                    pos = edge_pos.get(key, 999)
                    max_pos = max(max_pos, pos)
                if max_pos < best_K:
                    best_cycle = cycle
                    best_K = max_pos
    
    # Additional polish: try slightly smaller K values around best_K
    if time.time() < deadline - 3:
        for dk in range(1, 5):
            test_K = best_K - dk
            if test_K < 24:
                break
            if time.time() > deadline - 3:
                break
            for sv in range(N):
                if time.time() > deadline - 3:
                    break
                cycle = find_hamilton_cycle_k(test_K, deadline, start_vertex=sv)
                if cycle is not None:
                    max_pos = 0
                    for i in range(N):
                        u = cycle[i]
                        v = cycle[(i + 1) % N]
                        key = (min(u, v), max(u, v))
                        pos = edge_pos.get(key, 999)
                        max_pos = max(max_pos, pos)
                    if max_pos <= test_K:
                        best_cycle = cycle
                        best_K = max_pos
                        break
    
    return best_cycle
# EVOLVE-BLOCK-END
