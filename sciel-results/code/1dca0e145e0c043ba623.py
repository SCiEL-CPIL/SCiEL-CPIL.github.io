import time
import sys
import random
from itertools import combinations

# Precompute edge positions
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map: (min(u,v), max(u,v)) -> 1-based position
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def cycle_score(cycle):
    """Compute the max edge position for a Hamilton cycle."""
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        p = get_pos(u, v)
        if p > mx:
            mx = p
    return mx

def build_adj_threshold(threshold):
    """Build adjacency list using only edges with position <= threshold."""
    n = N
    adj = [set() for _ in range(n)]
    for (u, v), pos in edge_pos.items():
        if pos <= threshold:
            adj[u].add(v)
            adj[v].add(u)
    return adj

def find_hamilton_cycle(adj, n, deadline=None):
    """Find a Hamilton cycle using backtracking with strong pruning."""
    # Quick checks
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Check connectivity
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
    
    adj_list = [list(adj[v]) for v in range(n)]
    adj_set = [adj[v] for v in range(n)]
    
    path = [0]
    used = [False] * n
    used[0] = True
    
    def backtrack():
        if deadline is not None and time.time() > deadline:
            return None
        
        if len(path) == n:
            last = path[-1]
            if 0 in adj_set[last]:
                return path[:]
            return None
        
        last = path[-1]
        remaining = n - len(path)
        
        candidates = []
        for nxt in adj_list[last]:
            if not used[nxt]:
                avail = sum(1 for nb in adj_set[nxt] if not used[nb])
                if remaining == 1:
                    if 0 not in adj_set[nxt]:
                        continue
                    candidates.append((0, nxt))
                else:
                    candidates.append((avail, nxt))
        
        if not candidates:
            return None
        
        candidates.sort()
        
        for _, nxt in candidates:
            if deadline is not None and time.time() > deadline:
                return None
            
            used[nxt] = True
            path.append(nxt)
            
            if remaining == 1:
                if 0 in adj_set[nxt]:
                    result = path[:]
                    path.pop()
                    used[nxt] = False
                    return result
                path.pop()
                used[nxt] = False
                continue
            
            # Check for isolated unvisited vertices (only when close to deadline pressure)
            if deadline is not None and time.time() > deadline - 5:
                for v in range(n):
                    if not used[v]:
                        has_unvisited = any(not used[nb] for nb in adj_set[v])
                        if not has_unvisited:
                            path.pop()
                            used[nxt] = False
                            break
                else:
                    result = backtrack()
                    if result is not None:
                        return result
                path.pop()
                used[nxt] = False
            else:
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                used[nxt] = False
        
        return None
    
    return backtrack()

def local_search(cycle, n, deadline=None):
    """Improve cycle using 2-opt and Or-opt moves."""
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        
        # 2-opt: reverse a segment
        for i in range(n):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                if abs(i - j) == 1 or (i == 0 and j == n - 1):
                    continue
                
                new_cycle = best_cycle[:]
                seg = new_cycle[i:j+1]
                seg.reverse()
                new_cycle[i:j+1] = seg
                new_score = cycle_score(new_cycle)
                
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    improved = True
                    if deadline is not None and time.time() > deadline:
                        return best_cycle
        
        # Or-opt: move a segment of 2-3 vertices to another position
        for seg_len in [1, 2, 3]:
            for i in range(n):
                for j in range(n):
                    if j == i or j == (i - 1) % n or j == (i + seg_len) % n:
                        continue
                    # Extract segment starting at i with length seg_len
                    seg = []
                    for k in range(seg_len):
                        seg.append(best_cycle[(i + k) % n])
                    
                    # Remove segment from cycle
                    remaining = []
                    for k in range(n):
                        idx = (i + k) % n
                        if not any(idx == (i + s) % n for s in range(seg_len)):
                            remaining.append(best_cycle[idx])
                    
                    # Insert segment after position j in remaining
                    new_cycle = remaining[:j+1] + seg + remaining[j+1:]
                    
                    if len(new_cycle) != n:
                        continue
                    
                    new_score = cycle_score(new_cycle)
                    if new_score < best_score:
                        best_score = new_score
                        best_cycle = new_cycle
                        improved = True
                        if deadline is not None and time.time() > deadline:
                            return best_cycle
    
    return best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    n = N
    total_edges = n * (n - 1) // 2  # 276
    
    best_cycle = None
    best_score = total_edges + 1
    
    # Strategy: Try thresholds from low to high, find first valid cycle, then refine
    # Start with a reasonable threshold and increase
    
    # First, find any valid cycle with full graph
    adj_full = [set() for _ in range(n)]
    for (u, v) in EDGE_ORDER:
        adj_full[u].add(v)
        adj_full[v].add(u)
    
    test_cycle = find_hamilton_cycle(adj_full, n, deadline)
    if test_cycle is not None:
        best_cycle = test_cycle
        best_score = cycle_score(best_cycle)
    
    # Now try to find cycles with lower thresholds
    # Binary search for the minimum threshold
    lo, hi = 24, best_score if best_cycle else total_edges
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        adj = build_adj_threshold(mid)
        cycle = find_hamilton_cycle(adj, n, deadline)
        
        if cycle is not None:
            score = cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Refine the best cycle found
    if best_cycle is not None:
        if time.time() < deadline:
            refined = local_search(best_cycle, n, deadline)
            if refined is not None:
                rs = cycle_score(refined)
                if rs < best_score:
                    best_score = rs
                    best_cycle = refined
    
    # If we have time, try more random restarts
    if best_cycle is not None and time.time() < deadline - 5:
        rng = random.Random(42)
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            
            # Create a random perturbation of the best cycle
            perturbed = best_cycle[:]
            for _ in range(rng.randint(2, 5)):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            refined = local_search(perturbed, n, deadline)
            if refined is not None:
                rs = cycle_score(refined)
                if rs < best_score:
                    best_score = rs
                    best_cycle = refined
    
    # Fallback
    if best_cycle is None:
        best_cycle = list(range(n))
        best_score = cycle_score(best_cycle)
    
    return best_cycle
