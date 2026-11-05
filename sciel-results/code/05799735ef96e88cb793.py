"""Find a Hamilton cycle with minimized max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_N = 24

# Build the edge arrival lookup from the pinned edge order
_EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# edge_arrival[u][v] = 1-based index in edge_order
_edge_arrival = [[0] * _N for _ in range(_N)]
for idx, (u, v) in enumerate(_EDGE_ORDER, 1):
    _edge_arrival[u][v] = idx
    _edge_arrival[v][u] = idx

# Precompute sorted adjacency by arrival index
_sorted_adj = [[] for _ in range(_N)]
for i in range(_N):
    for j in range(_N):
        if i != j:
            _sorted_adj[i].append((_edge_arrival[i][j], j))
    _sorted_adj[i].sort()


def _cycle_score(cycle):
    """Score = max one-based arrival index of edges in the cycle (with closing edge)."""
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if _edge_arrival[u][v] > mx:
            mx = _edge_arrival[u][v]
    return mx


def _build_adj(k):
    """Build adjacency list and mask using only edges with arrival index <= k."""
    adj = [[] for _ in range(_N)]
    adj_mask = [0] * _N
    for i in range(_N):
        for j in range(i + 1, _N):
            if _edge_arrival[i][j] <= k:
                adj[i].append(j)
                adj[j].append(i)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj, adj_mask


def _find_cycle_dfs(k, deadline=None):
    """
    Find a Hamilton cycle using only edges with arrival index <= k.
    Uses DFS with strong pruning: degree checks, connectivity, and articulation.
    """
    if deadline is None:
        deadline = time.time() + 10.0
    
    adj, adj_mask = _build_adj(k)
    FULL = (1 << _N) - 1
    
    # Pre-check: min degree >= 2
    for i in range(_N):
        if len(adj[i]) < 2:
            return None
    
    # Pre-check: connectivity
    visited = [False] * _N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    if count < _N:
        return None
    
    # DFS with backtracking
    path = [0]
    visited_mask = 1  # vertex 0 is visited
    found = False
    
    def dfs(current):
        nonlocal found, visited_mask
        
        if time.time() > deadline:
            return
        
        if len(path) == _N:
            if _edge_arrival[current][0] <= k:
                found = True
            return
        
        # Get candidates sorted by degree (Warnsdorff-like)
        candidates = []
        for v in adj[current]:
            if visited_mask & (1 << v):
                continue
            if len(path) == _N - 1:
                # Last vertex must connect to start
                if _edge_arrival[v][0] <= k:
                    candidates.append(v)
            else:
                candidates.append(v)
        
        if not candidates:
            return
        
        # Sort by degree in remaining graph (ascending - Warnsdorff)
        remaining = FULL & ~visited_mask
        scored = []
        for v in candidates:
            next_rem = remaining & ~(1 << v)
            deg = bin(adj_mask[v] & next_rem).count('1')
            scored.append((deg, v))
        scored.sort()
        
        for deg, v in scored:
            if found or time.time() > deadline:
                return
            
            new_mask = visited_mask | (1 << v)
            new_remaining = FULL & ~new_mask
            
            # Pruning: check if any unvisited vertex is isolated
            valid = True
            r = new_remaining
            while r:
                bit = r & -r
                u = bit.bit_length() - 1
                r -= bit
                # u has no unvisited neighbors
                if (adj_mask[u] & new_remaining) == 0:
                    # Must have >= 2 neighbors in visited set
                    if bin(adj_mask[u] & new_mask).count('1') < 2:
                        valid = False
                        break
            if not valid:
                continue
            
            # Pruning: check connectivity of remaining unvisited vertices
            # plus the path endpoints (0 and current)
            # Actually, simpler: check if new_remaining is connected
            # (including connections to path)
            # For speed, do a quick BFS on new_remaining
            if new_remaining:
                start_v = (new_remaining & -new_remaining).bit_length() - 1
                vis = 0
                stack2 = [start_v]
                vis |= (1 << start_v)
                while stack2:
                    u2 = stack2.pop()
                    # Neighbors of u2 in new_remaining
                    nbrs = adj_mask[u2] & new_remaining
                    while nbrs:
                        bit2 = nbrs & -nbrs
                        w = bit2.bit_length() - 1
                        nbrs -= bit2
                        if not (vis & (1 << w)):
                            vis |= (1 << w)
                            stack2.append(w)
                # Check all of new_remaining is reached
                if vis != new_remaining:
                    continue
            
            # Commit
            path.append(v)
            visited_mask = new_mask
            dfs(v)
            path.pop()
            visited_mask = new_mask & ~(1 << v)
            if found:
                return
    
    dfs(0)
    if found:
        return path[:]
    return None


def _local_search_2opt(cycle, deadline=None):
    """Improve a Hamilton cycle using 2-opt moves to minimize max edge arrival."""
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best)
    
    if deadline is None:
        deadline = time.time() + 5.0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment i+1..j
                new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_score = _cycle_score(new_cycle)
                if new_score < best_score:
                    best = new_cycle
                    best_score = new_score
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    return best, best_score


def _greedy_construct(k, rng=None):
    """
    Greedy construction: build a path by always picking the lowest-arrival 
    unvisited neighbor that doesn't disconnect the remaining graph.
    """
    if rng is None:
        rng = random.Random(42)
    
    adj, adj_mask = _build_adj(k)
    FULL = (1 << _N) - 1
    
    path = [0]
    visited_mask = 1
    
    for step in range(1, _N):
        current = path[-1]
        remaining = FULL & ~visited_mask
        
        # Get candidates: unvisited neighbors of current
        candidates = []
        for v in adj[current]:
            if visited_mask & (1 << v):
                continue
            if step == _N - 1:
                # Last vertex must connect to 0
                if _edge_arrival[v][0] <= k:
                    candidates.append(v)
            else:
                candidates.append(v)
        
        if not candidates:
            return None
        
        # Sort by arrival index (prefer earlier edges)
        candidates.sort(key=lambda v: _edge_arrival[current][v])
        
        # Try candidates in order with connectivity check
        for v in candidates:
            new_mask = visited_mask | (1 << v)
            new_remaining = FULL & ~new_mask
            
            if new_remaining:
                # Check connectivity of new_remaining
                start_v = (new_remaining & -new_remaining).bit_length() - 1
                vis = 0
                stack2 = [start_v]
                vis |= (1 << start_v)
                while stack2:
                    u2 = stack2.pop()
                    nbrs = adj_mask[u2] & new_remaining
                    while nbrs:
                        bit2 = nbrs & -nbrs
                        w = bit2.bit_length() - 1
                        nbrs -= bit2
                        if not (vis & (1 << w)):
                            vis |= (1 << w)
                            stack2.append(w)
                if vis != new_remaining:
                    continue
            
            path.append(v)
            visited_mask = new_mask
            break
        else:
            return None
    
    # Check closing edge
    if _edge_arrival[path[-1]][0] <= k:
        return path
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # Start with a reasonable baseline
    best_cycle = list(range(_N))
    best_score = _cycle_score(best_cycle)
    
    # Strategy: Binary search for minimal feasible k, then refine
    # Edge indices range from 1 to 276
    
    # First, find a feasible k using binary search
    lo, hi = 2, 276
    feasible_k = None
    
    # Time budget for binary search: ~40s
    bs_deadline = start_time + 40.0
    
    while lo <= hi and time.time() < bs_deadline:
        mid = (lo + hi) // 2
        
        # Quick feasibility: min degree >= 2
        min_deg = _N
        for i in range(_N):
            deg = 0
            for j in range(_N):
                if i != j and _edge_arrival[i][j] <= mid:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
        if min_deg < 2:
            lo = mid + 1
            continue
        
        # Try to find a cycle with limited time
        remaining_time = bs_deadline - time.time()
        if remaining_time < 0.5:
            break
        
        timeout = min(3.0, remaining_time * 0.3)
        result = _find_cycle_dfs(mid, deadline=time.time() + timeout)
        
        if result:
            feasible_k = mid
            score = _cycle_score(result)
            if score < best_score:
                best_cycle = result
                best_score = score
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If we found a feasible k, try to find a cycle at that k with more time
    if feasible_k is not None and time.time() < deadline - 10.0:
        # Try multiple times with different strategies
        for attempt in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Try DFS
            cycle = _find_cycle_dfs(feasible_k, deadline=time.time() + 5.0)
            if cycle:
                score = _cycle_score(cycle)
                if score < best_score:
                    best_cycle = cycle
                    best_score = score
                break
            
            # Try greedy
            rng = random.Random(attempt * 100 + 7)
            cycle = _greedy_construct(feasible_k, rng=rng)
            if cycle:
                score = _cycle_score(cycle)
                if score < best_score:
                    best_cycle = cycle
                    best_score = score
                break
    
    # If we still have time, try to improve with local search
    if time.time() < deadline - 2.0:
        final_cycle, final_score = _local_search_2opt(
            best_cycle, deadline=deadline - 1.0
        )
        if final_score < best_score:
            best_cycle = final_cycle
            best_score = final_score
    
    # If best_score is still high, try random restarts with local search
    if best_score > 150 and time.time() < deadline - 3.0:
        rng = random.Random(123)
        ls_deadline = deadline - 1.0
        for restart in range(20):
            if time.time() > ls_deadline:
                break
            perm = list(range(_N))
            rng.shuffle(perm)
            improved_cycle, improved_score = _local_search_2opt(
                perm, deadline=min(ls_deadline, time.time() + 1.5)
            )
            if improved_score < best_score:
                best_cycle = improved_cycle
                best_score = improved_score
    
    return best_cycle

# EVOLVE-BLOCK-END
