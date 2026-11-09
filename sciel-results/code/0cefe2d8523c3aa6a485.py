import time
import sys
from collections import defaultdict

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1


def compute_cycle_score(cycle):
    """Compute the score (max edge position) of a Hamilton cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        a, b = min(u, v), max(u, v)
        p = edge_pos[(a, b)]
        if p > max_pos:
            max_pos = p
    return max_pos


def two_opt_improve(cycle, deadline):
    """Apply 2-opt local search to reduce the max edge arrival index."""
    n = len(cycle)
    current_score = compute_cycle_score(cycle)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try all pairs of edges to swap
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            for j in range(i + 1, n - 1):
                # Skip if edges are adjacent
                if j == i + 1:
                    continue
                
                # Try reversing the segment between i and j
                # Original: ... A - B ... C - D ...
                # New: ... A - C ... B - D ...
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                
                new_score = compute_cycle_score(new_cycle)
                
                if new_score < current_score:
                    cycle = new_cycle
                    current_score = new_score
                    improved = True
        
        # Also try 3-opt: reverse segment and check
        if time.time() < deadline:
            for i in range(n - 2):
                if time.time() > deadline:
                    break
                for j in range(i + 2, n - 1):
                    # Reverse segment from i to j
                    new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                    new_score = compute_cycle_score(new_cycle)
                    if new_score < current_score:
                        cycle = new_cycle
                        current_score = new_score
                        improved = True
    
    return cycle
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1


def find_hamilton_cycle_fast(adj, n, deadline):
    """
    Fast Hamilton cycle finder with improved pruning.
    Returns the list of vertices in the cycle or None if not found.
    """
    # Quick checks
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Connectivity check
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 0
    while stack:
        u = stack.pop()
        count += 1
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                stack.append(w)
    if count != n:
        return None

    # Pre-compute adjacency sets for O(1) lookup
    adj_sets = [set(adj[v]) for v in range(n)]
    
    # For each vertex, precompute which neighbors connect to 0 (for closing edge check)
    # A vertex v can be the last in the path if v connects to 0
    can_close = [1 if 0 in adj_sets[v] else 0 for v in range(n)]
    
    path = [0]
    used = [False] * n
    used[0] = True
    
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            # Check closing edge
            last = path[-1]
            if 0 in adj_sets[last]:
                return path[:]
            return None
        
        last = path[-1]
        
        # Heuristic: If only 1 vertex left, it must connect to last and 0
        if len(path) == n - 1:
            remaining = -1
            for v in range(n):
                if not used[v]:
                    remaining = v
                    break
            if remaining != -1:
                if remaining in adj_sets[last] and 0 in adj_sets[remaining]:
                    path.append(remaining)
                    return path[:]
            return None

        # Try neighbors with improved heuristic
        neighbors = []
        for nxt in adj[last]:
            if not used[nxt]:
                # Count unvisited neighbors of nxt (excluding last and 0 for the count)
                deg = 0
                for w in adj[nxt]:
                    if not used[w] and w != last and w != 0:
                        deg += 1
                # Prioritize vertices that can close the cycle (connect to 0)
                # and have fewer unvisited neighbors (Warnsdorff)
                neighbors.append((deg, not can_close[nxt], nxt))
        
        # Sort by degree (ascending), then prefer vertices that CAN close
        neighbors.sort(key=lambda x: (x[0], x[1]))
        
        for deg, _, nxt in neighbors:
            if used[nxt]:
                continue
            
            # Critical pruning: After choosing nxt, check if any unvisited vertex 
            # (other than nxt) becomes isolated from the remaining subgraph
            # The remaining path must go through all unvisited vertices and end at 0
            # For each unvisited w (w != nxt), w must have a neighbor in:
            #   (Unvisited \ {nxt}) or {nxt} or {0}
            
            is_dead_end = False
            for w in range(n):
                if not used[w] and w != nxt:
                    has_connection = False
                    if nxt in adj_sets[w]:
                        has_connection = True
                    elif 0 in adj_sets[w]:
                        has_connection = True
                    else:
                        for nb in adj[w]:
                            if not used[nb] and nb != nxt:
                                has_connection = True
                                break
                    if not has_connection:
                        is_dead_end = True
                        break
            
            if is_dead_end:
                continue

            used[nxt] = True
            path.append(nxt)
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            used[nxt] = False
        
        return None

    return backtrack()


def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = N
    total_edges = n * (n - 1) // 2
    
    # Binary search for the minimum threshold T
    lo, hi = 24, total_edges
    best_cycle = None
    best_score = total_edges + 1
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build subgraph with edges at position <= mid
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj[u].append(v)
                adj[v].append(u)
        
        # Check for Hamilton cycle
        cycle = find_hamilton_cycle_fast(adj, n, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        # Apply 2-opt improvement to reduce the actual score
        improved_cycle = two_opt_improve(best_cycle, deadline)
        improved_score = compute_cycle_score(improved_cycle)
        
        if improved_score < best_score:
            best_cycle = improved_cycle
            best_score = improved_score
        
        return best_cycle
    
    # Fallback: try full graph
    if time.time() < deadline:
        adj = [[] for _ in range(n)]
        for (u, v), pos in edge_pos.items():
            adj[u].append(v)
            adj[v].append(u)
        
        cycle = find_hamilton_cycle_fast(adj, n, start_time + 5)
        if cycle:
            # Apply 2-opt improvement
            improved_cycle = two_opt_improve(cycle, deadline)
            return improved_cycle
            
    # Final fallback
    return list(range(n))


if __name__ == "__main__":
    c = construct()
    max_pos = compute_cycle_score(c)
    print(f"Cycle: {c}")
    print(f"Score: {max_pos}")
