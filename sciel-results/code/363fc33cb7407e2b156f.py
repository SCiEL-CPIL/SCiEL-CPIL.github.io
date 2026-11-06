"""Snake-in-the-box in Q_9: A* Backtracking Solver with Matching Heuristic."""
import time
import random
import sys
import heapq

N = 9
MASK = (1 << N) - 1
SIZE = 1 << N

# Precompute neighbors
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Precompute neighbor masks for fast intersection
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for nb in NEIGHBORS[v]:
        m |= (1 << nb)
    NEIGHBOR_MASK[v] = m

def _estimate_remaining(cur, used_mask, blocked_mask, depth):
    """
    Estimate how many more vertices can be added after `cur`.
    Uses a greedy matching-like heuristic: count available neighbors and 
    estimate based on branching factor decay.
    """
    # Available neighbors of cur: not used, not blocked
    avail_mask = NEIGHBOR_MASK[cur] & ~used_mask & ~blocked_mask
    avail_count = bin(avail_mask).count('1')
    
    if avail_count == 0:
        return 0
    
    # Simple heuristic: assume each step reduces options by ~2
    # This is very rough but guides the search
    est = 0
    remaining = avail_count
    while remaining > 0 and est < 200:
        est += 1
        remaining = max(0, remaining - 2)
    
    return est

def _solve_a_star(start, deadline, rng):
    """
    A*-guided DFS with Warnsdorff's rule and matching-based heuristic.
    State: (current_vertex, used_mask, blocked_mask, path)
    blocked_mask: vertices that are neighbors of any vertex in path[:-1]
    """
    best_path = [start]
    best_len = 1
    
    sys.setrecursionlimit(10000)
    
    def dfs(cur, used_mask, blocked_mask, path, depth):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Compute new_blocked: after adding next vertex, blocked becomes 
        # neighbors of all vertices in current path (i.e., blocked | NEIGHBORS[cur])
        new_blocked = blocked_mask | NEIGHBOR_MASK[cur]
        
        # Find candidates: neighbors of cur that are not used and not blocked
        candidates = []
        for nb in NEIGHBORS[cur]:
            if (used_mask >> nb) & 1:
                continue
            if (blocked_mask >> nb) & 1:
                continue
            
            # Warnsdorff + A* heuristic score
            # Count valid onward moves from nb
            onward_mask = NEIGHBOR_MASK[nb] & ~(used_mask | (1 << nb)) & ~new_blocked
            onward_count = bin(onward_mask).count('1')
            
            # Heuristic: prefer candidates with fewer onward moves (Warnsdorff)
            # but also consider the estimated total remaining
            # Lower score = more promising
            score = onward_count * 10 - _estimate_remaining(nb, used_mask | (1 << nb), new_blocked, depth + 1)
            
            candidates.append((score, onward_count, nb))
        
        if not candidates:
            return
        
        # Sort by score (primary), then by onward_count (secondary)
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        # Try candidates in order
        for score, onward_count, nb in candidates:
            path.append(nb)
            new_used = used_mask | (1 << nb)
            
            dfs(nb, new_used, new_blocked, path, depth + 1)
            
            path.pop()
            
            if time.time() > deadline:
                return
    
    path = [start]
    used_mask = 1 << start
    blocked_mask = 0  # No vertices before the first, so nothing is blocked
    
    dfs(start, used_mask, blocked_mask, path, 0)
    
    return best_path

def _greedy_extend(start, deadline, rng):
    """
    Fast greedy extension using Warnsdorff's rule, no backtracking.
    Used to quickly get a good initial solution.
    """
    path = [start]
    used_mask = 1 << start
    blocked_mask = 0
    cur = start
    
    while time.time() < deadline:
        new_blocked = blocked_mask | NEIGHBOR_MASK[cur]
        
        candidates = []
        for nb in NEIGHBORS[cur]:
            if (used_mask >> nb) & 1:
                continue
            if (blocked_mask >> nb) & 1:
                continue
            
            onward_mask = NEIGHBOR_MASK[nb] & ~(used_mask | (1 << nb)) & ~new_blocked
            onward_count = bin(onward_mask).count('1')
            
            candidates.append((onward_count, nb))
        
        if not candidates:
            break
        
        candidates.sort()
        _, nb = candidates[0]
        
        path.append(nb)
        used_mask |= (1 << nb)
        blocked_mask = new_blocked
        cur = nb
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Quick greedy extensions to get a good baseline
    greedy_deadline = start_time + 5
    starts = [0] + [rng.randint(0, SIZE - 1) for _ in range(20)]
    
    for s in starts:
        if time.time() > greedy_deadline:
            break
        path = _greedy_extend(s, min(greedy_deadline, time.time() + 0.5), rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    # Phase 2: A* backtracking search from promising starts
    # Use the best greedy result's start and nearby vertices
    search_deadline = deadline
    
    # Try A* from multiple starts, allocating time proportionally
    num_searches = 15
    time_per_search = (search_deadline - time.time()) / num_searches if time.time() < search_deadline else 0
    
    if time_per_search > 0.1:
        # Use starts from the best greedy path and random ones
        search_starts = list(best_path[:min(5, len(best_path))]) + [rng.randint(0, SIZE - 1) for _ in range(num_searches - min(5, len(best_path)))]
        
        for s in search_starts:
            if time.time() > search_deadline:
                break
            
            sub_deadline = min(search_deadline, time.time() + time_per_search)
            path = _solve_a_star(s, sub_deadline, rng)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    return best_path
