"""Find Hamilton cycle minimizing max edge arrival index on 24 vertices."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Build edge index map: (min(a,b), max(a,b)) -> one-based arrival index
edge_idx = {}
for i, (a, b) in enumerate(EDGE_ORDER):
    key = (min(a, b), max(a, b))
    edge_idx[key] = i + 1  # one-based

def get_edge_index(a, b):
    return edge_idx[(min(a, b), max(a, b))]

def find_hamilton_cycle_in_subgraph(allowed_edges, start=0):
    """Find a Hamilton cycle using only edges in allowed_edges set.
    Returns list of vertices or None."""
    n = N
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (a, b) in allowed_edges:
        adj[a].add(b)
        adj[b].add(a)
    
    # Backtracking with bitmask
    visited = 1 << start
    path = [start]
    
    def backtrack():
        if len(path) == n:
            # Check if we can return to start
            if start in adj[path[-1]]:
                return True
            return False
        last = path[-1]
        for nxt in sorted(adj[last]):
            if not (visited & (1 << nxt)):
                path.append(nxt)
                visited |= (1 << nxt)
                if backtrack():
                    return True
                path.pop()
                visited &= ~(1 << nxt)
        return False
    
    # Try different starts if needed, but start with 0
    if backtrack():
        return path
    return None

def greedy_construct(threshold):
    """Greedy: build cycle using only edges with arrival index <= threshold."""
    n = N
    # Get allowed edges
    allowed = set()
    for (a, b), idx in edge_idx.items():
        if idx <= threshold:
            allowed.add((a, b))
    
    adj = [set() for _ in range(n)]
    for (a, b) in allowed:
        adj[a].add(b)
        adj[b].add(a)
    
    # Greedy: start at vertex 0, always pick neighbor with fewest remaining options
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        last = path[-1]
        candidates = [v for v in adj[last] if not visited[v]]
        if not candidates:
            return None
        # Pick the one with fewest unvisited neighbors (Warnsdorff-like)
        candidates.sort(key=lambda v: sum(1 for u in adj[v] if not visited[u]))
        nxt = candidates[0]
        path.append(nxt)
        visited[nxt] = True
    
    # Check closing edge
    if (min(path[-1], 0), max(path[-1], 0)) not in allowed:
        return None
    return path

def two_opt_improve(cycle, threshold):
    """Improve cycle using 2-opt while staying within threshold."""
    n = N
    cycle = list(cycle)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Remove edges (cycle[i], cycle[i+1]) and (cycle[j], cycle[j+1])
                # Add edges (cycle[i], cycle[j]) and (cycle[i+1], cycle[j+1])
                a, b = cycle[i], cycle[i + 1]
                c, d = cycle[j], cycle[j + 1] if j + 1 < n else cycle[0]
                
                # Check new edges exist in allowed set
                if (min(a, c), max(a, c)) not in edge_idx or edge_idx[(min(a, c), max(a, c))] > threshold:
                    continue
                if (min(b, d), max(b, d)) not in edge_idx or edge_idx[(min(b, d), max(b, d))] > threshold:
                    continue
                
                # Check we're not creating a 2-cycle
                if abs(i - j) == 1:
                    continue
                if i == 0 and j == n - 1:
                    continue
                
                # Reverse segment
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                
                # Verify all edges in new cycle are within threshold
                valid = True
                for k in range(n):
                    u, v = new_cycle[k], new_cycle[(k+1) % n]
                    key = (min(u, v), max(u, v))
                    if key not in edge_idx or edge_idx[key] > threshold:
                        valid = False
                        break
                if valid:
                    cycle = new_cycle
                    improved = True
                    break
            if improved:
                break
    return cycle

def find_best_cycle():
    """Binary search on threshold, then find and improve cycle."""
    t0 = time.time()
    deadline = 75.0  # safety margin
    
    # First, get a good initial solution with greedy + 2-opt at a high threshold
    # Start with threshold = max index
    max_idx = max(edge_idx.values())
    
    # Try to find a cycle with progressively lower thresholds
    best_cycle = None
    best_score = max_idx
    
    # Start from a reasonable threshold and binary search
    lo, hi = 1, max_idx
    
    # First get any valid cycle
    init_cycle = greedy_construct(max_idx)
    if init_cycle is None:
        # Fallback: identity
        init_cycle = list(range(N))
    
    # Improve with 2-opt
    init_cycle = two_opt_improve(init_cycle, max_idx)
    init_score = max(get_edge_index(cycle[i], cycle[(i+1)%N]) for i in range(N))
    best_cycle = init_cycle
    best_score = init_score
    
    # Binary search for minimum threshold
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = greedy_construct(mid)
        if cycle is not None:
            cycle = two_opt_improve(cycle, mid)
            # Verify
            valid = True
            for i in range(N):
                u, v = cycle[i], cycle[(i+1) % N]
                if get_edge_index(u, v) > mid:
                    valid = False
                    break
            if valid:
                score = max(get_edge_index(cycle[i], cycle[(i+1)%N]) for i in range(N))
                if score < best_score:
                    best_score = score
                    best_cycle = cycle
                hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Final polish: try to reduce score further with local search
    if time.time() < deadline:
        # Try 2-opt with progressively tighter thresholds
        for thresh in range(best_score - 1, lo - 1, -1):
            if time.time() > deadline:
                break
            # Try to find a cycle within this threshold
            for attempt in range(20):
                if time.time() > deadline:
                    break
                # Randomized greedy
                random.seed(attempt)
                cycle = None
                # Try different starting vertices
                start = attempt % N
                visited = [False] * N
                path = [start]
                visited[start] = True
                adj_thresh = [set() for _ in range(N)]
                for (a, b), idx in edge_idx.items():
                    if idx <= thresh:
                        adj_thresh[a].add(b)
                        adj_thresh[b].add(a)
                
                for _ in range(N - 1):
                    last = path[-1]
                    candidates = [v for v in adj_thresh[last] if not visited[v]]
                    if not candidates:
                        break
                    # Random selection with preference for low-degree
                    candidates.sort(key=lambda v: (sum(1 for u in adj_thresh[v] if not visited[u]), random.random()))
                    nxt = candidates[0]
                    path.append(nxt)
                    visited[nxt] = True
                
                if len(path) == N:
                    if (min(path[-1], start), max(path[-1], start)) in {(min(a,b),max(a,b)) for (a,b),idx in edge_idx.items() if idx <= thresh}:
                        # Verify full cycle
                        valid = True
                        for i in range(N):
                            u, v = path[i], path[(i+1) % N]
                            if get_edge_index(u, v) > thresh:
                                valid = False
                                break
                        if valid:
                            cycle = path
                            break
                
                if cycle is not None:
                    cycle = two_opt_improve(cycle, thresh)
                    score = max(get_edge_index(cycle[i], cycle[(i+1)%N]) for i in range(N))
                    if score < best_score:
                        best_score = score
                        best_cycle = cycle
                    break
    
    return best_cycle if best_cycle is not None else list(range(N))

def construct():
    t0 = time.time()
    deadline = 75.0
    
    # Quick check: if we have time, do the full search
    if time.time() < deadline:
        cycle = find_best_cycle()
    else:
        cycle = list(range(N))
    
    # Ensure valid permutation
    if not isinstance(cycle, list) or len(cycle) != N or sorted(cycle) != list(range(N)):
        cycle = list(range(N))
    
    return cycle
# EVOLVE-BLOCK-END
