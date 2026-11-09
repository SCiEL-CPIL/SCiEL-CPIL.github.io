"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edge_order = P['edge_order']
    num_edges = len(edge_order)
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute edge positions
    edge_pos = {}
    for i, (u, v) in enumerate(edge_order):
        edge_pos[(u, v)] = i
        edge_pos[(v, u)] = i

    def get_cycle_score(cycle):
        """Return the one-based max edge index in the cycle."""
        max_idx = 0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            idx = edge_pos[(u, v)]
            if idx + 1 > max_idx:
                max_idx = idx + 1
        return max_idx

    def two_opt(cycle, threshold):
        """
        Try to improve the cycle by 2-opt moves, keeping all edges < threshold.
        Returns improved cycle or original if no improvement.
        """
        cycle = list(cycle)
        improved = True
        while improved:
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    a, b = cycle[i], cycle[i + 1]
                    c, d = cycle[j], cycle[j + 1]
                    if a == c or a == d or b == c or b == d:
                        continue
                    old_score = max(edge_pos[(a, b)], edge_pos[(c, d)])
                    new_score = max(edge_pos[(a, c)], edge_pos[(b, d)])
                    if new_score < old_score and new_score < threshold:
                        cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
                        improved = True
        return cycle

    def greedy_build(threshold):
        """
        Build a Hamiltonian path greedily using Warnsdorff's rule (prefer
        neighbors with fewer onward options), then try to close it.
        """
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
        
        for start in range(n):
            path = [start]
            visited = [False] * n
            visited[start] = True
            
            for _ in range(n - 1):
                last = path[-1]
                candidates = [v for v in adj[last] if not visited[v]]
                if not candidates:
                    break
                # Warnsdorff: pick the candidate with fewest unvisited neighbors
                candidates.sort(key=lambda v: (sum(1 for w in adj[v] if not visited[w]), edge_pos[(last, v)]))
                path.append(candidates[0])
                visited[candidates[0]] = True
            
            if len(path) == n:
                last = path[-1]
                if edge_pos.get((last, start), float('inf')) < threshold:
                    return path
        
        return None

    def find_hamiltonian_cycle(threshold, time_limit=None):
        """
        Find a Hamiltonian cycle using only edges with position < threshold.
        Returns the cycle (list of vertices) or None if not found.
        Uses greedy first, then backtracking with strong pruning.
        """
        if time_limit is None:
            time_limit = DEADLINE - 5

        # Try greedy first (fast)
        cycle = greedy_build(threshold)
        if cycle:
            return cycle
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(threshold):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            adj[u].sort(key=lambda v: edge_pos[(u, v)])
        
        # Quick feasibility check: every vertex must have degree >= 2
        for u in range(n):
            if len(adj[u]) < 2:
                return None
        
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        # Track how many times each vertex is "used" in the path (0, 1, or 2)
        used_count = [0] * n
        used_count[start] = 1  # start has one edge in the path (the closing one)
        
        def is_feasible():
            """Check if the partial path can still be extended to a Hamilton cycle."""
            for u in range(n):
                if not visited[u]:
                    # Unvisited vertex must have at least 2 unvisited neighbors
                    # (it needs two edges in the final cycle)
                    unvisited_neighbors = sum(1 for v in adj[u] if not visited[v])
                    # It also needs to connect to visited vertices for the cycle
                    if unvisited_neighbors < 1:
                        return False
                else:
                    # Visited vertex (not start, not last): must have exactly 2 used edges
                    if u != start and u != path[-1]:
                        if used_count[u] < 2:
                            # Can still get more edges
                            pass
            # Check that no unvisited vertex is isolated from the rest
            for u in range(n):
                if not visited[u]:
                    total_unvisited_neighbors = sum(1 for v in adj[u] if not visited[v])
                    if total_unvisited_neighbors == 0:
                        return False
            return True

        def backtrack(depth):
            if time.time() > time_limit:
                return False
            if len(path) == n:
                last = path[-1]
                if edge_pos.get((last, start), float('inf')) < threshold:
                    return True
                return False
            
            last = path[-1]
            # Sort candidates: prefer those with fewer onward options (Warnsdorff)
            candidates = []
            for next_v in adj[last]:
                if not visited[next_v]:
                    if next_v == start:
                        continue
                    # Count unvisited neighbors of next_v (excluding last)
                    onward = sum(1 for w in adj[next_v] if not visited[w] and w != last)
                    candidates.append((onward, edge_pos[(last, next_v)], next_v))
            
            candidates.sort()
            
            for _, _, next_v in candidates:
                path.append(next_v)
                visited[next_v] = True
                used_count[next_v] = 1
                
                if is_feasible():
                    if backtrack(depth + 1):
                        return True
                
                path.pop()
                visited[next_v] = False
                used_count[next_v] = 0
            return False
        
        if backtrack(0):
            return path
        return None

    # Binary search for the minimum threshold
    lo = n
    hi = num_edges
    best_cycle = None
    best_threshold = num_edges
    
    # Get a fallback cycle with full edge set
    full_cycle = find_hamiltonian_cycle(num_edges)
    if full_cycle:
        best_cycle = full_cycle
        best_threshold = num_edges
    else:
        return list(range(n))
        
    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        if time.time() > DEADLINE - 10:
            break
        
        cycle = find_hamiltonian_cycle(mid)
        if cycle:
            best_cycle = cycle
            best_threshold = mid
            hi = mid
        else:
            lo = mid + 1
            
    # Polish: apply 2-opt to improve the specific cycle
    if best_cycle:
        improved = two_opt(best_cycle, best_threshold)
        if get_cycle_score(improved) < get_cycle_score(best_cycle):
            best_cycle = improved
        
        # Try to find an even better threshold by checking just below
        for t in range(best_threshold - 1, max(n - 1, best_threshold - 8), -1):
            if time.time() > DEADLINE - 5:
                break
            cycle = find_hamiltonian_cycle(t)
            if cycle:
                improved = two_opt(cycle, t)
                if get_cycle_score(improved) < get_cycle_score(best_cycle):
                    best_cycle = improved
                    best_threshold = t
                break
    
    if best_cycle:
        return best_cycle
    else:
        return list(range(n))

# EVOLVE-BLOCK-END
