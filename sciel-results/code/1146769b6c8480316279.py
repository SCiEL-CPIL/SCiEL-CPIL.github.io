"""Snake-in-the-box Q_9: systematic DFS with aggressive pruning and time guard."""
import time
import sys
import numpy as np

N = 9
TOTAL_VERTICES = 1 << N  # 512

# Precompute bit masks
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbors for each vertex (list of neighbors)
# neighbors[v] = list of 9 neighbors of v
NEIGHBORS = None

def _init_neighbors():
    global NEIGHBORS
    NEIGHBORS = [[] for _ in range(TOTAL_VERTICES)]
    for v in range(TOTAL_VERTICES):
        for b in range(N):
            NEIGHBORS[v].append(v ^ (1 << b))

_init_neighbors()

def _greedy_seed():
    """Generate a good initial snake using Warnsdorff-like heuristic."""
    path = [0]
    used = [False] * TOTAL_VERTICES
    used[0] = True
    
    while True:
        cur = path[-1]
        best_next = -1
        best_degree = -1
        best_v = -1
        
        for b in range(N):
            v = cur ^ (1 << b)
            if used[v]:
                continue
            # Check if v is adjacent to any non-consecutive vertex in path
            valid = True
            for i in range(len(path) - 2):  # exclude last (consecutive)
                if (v ^ path[i]) & ((v ^ path[i]) - 1) == 0:  # popcount == 1
                    valid = False
                    break
            if not valid:
                continue
            # Count available neighbors of v (degree heuristic)
            deg = 0
            for nb in NEIGHBORS[v]:
                if not used[nb]:
                    # Check if nb would be valid (not adjacent to non-consecutive)
                    # For speed, just count unused neighbors
                    deg += 1
            if deg > best_degree:
                best_degree = deg
                best_next = v
        
        if best_next == -1:
            break
        
        path.append(best_next)
        used[best_next] = True
    
    return path

def _is_valid_snake(path):
    """Verify that path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = [False] * TOTAL_VERTICES
    for v in path:
        if v < 0 or v >= TOTAL_VERTICES:
            return False
        if used[v]:
            return False
        used[v] = True
    
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:  # popcount != 1
            return False
    
    for i in range(n):
        for j in range(i + 2, n):
            diff = path[i] ^ path[j]
            if diff & (diff - 1) == 0:  # popcount == 1
                return False
    return True

def _dfs_extend(path, used, deadline, best_path_ref):
    """
    DFS to extend the current path. Returns the longest path found.
    path: list of vertices
    used: set of used vertices
    deadline: time.time() deadline
    best_path_ref: list to store best path found
    """
    cur = path[-1]
    n_path = len(path)
    
    # Try all possible next vertices
    for b in range(N):
        v = cur ^ (1 << b)
        if v in used:
            continue
        
        # Check induced constraint: v must not be adjacent to any path[i] for i < n_path-2
        valid = True
        for i in range(n_path - 2):
            diff = v ^ path[i]
            if diff & (diff - 1) == 0:  # Hamming distance 1
                valid = False
                break
        if not valid:
            continue
        
        # Extend
        path.append(v)
        used.add(v)
        
        if len(path) > len(best_path_ref[0]):
            best_path_ref[0] = list(path)
        
        # Recurse
        if time.time() < deadline:
            _dfs_extend(path, used, deadline, best_path_ref)
        
        path.pop()
        used.remove(v)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 110s out of 150s
    
    # Generate initial seed
    seed_path = _greedy_seed()
    best_path = list(seed_path)
    
    if time.time() > deadline:
        return best_path
    
    used = set(seed_path)
    
    # Try to extend from the seed using DFS with backtracking
    # We'll do iterative deepening / restarts to explore different branches
    
    # First, try to extend the seed greedily with backtracking
    best_path_ref = [best_path]
    
    # DFS from the end of the seed
    _dfs_extend(list(seed_path), used, deadline, best_path_ref)
    
    if time.time() > deadline:
        return best_path_ref[0]
    
    best_path = best_path_ref[0]
    
    # Try multiple restarts with different starting vertices and orderings
    # to find longer snakes
    
    # Precompute a good ordering for bit flips
    # Try different starting points
    for start_v in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if time.time() > deadline - 5:
            break
        
        # Generate a greedy snake from start_v
        path = [start_v]
        used_local = set([start_v])
        
        while True:
            cur = path[-1]
            best_next = -1
            best_degree = -1
            
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used_local:
                    continue
                # Check induced constraint
                valid = True
                for i in range(len(path) - 2):
                    diff = v ^ path[i]
                    if diff & (diff - 1) == 0:
                        valid = False
                        break
                if not valid:
                    continue
                deg = sum(1 for nb in NEIGHBORS[v] if nb not in used_local)
                if deg > best_degree:
                    best_degree = deg
                    best_next = v
            
            if best_next == -1:
                break
            
            path.append(best_next)
            used_local.add(best_next)
        
        if len(path) > len(best_path):
            best_path = list(path)
        
        # Try DFS extension from this new seed
        if len(path) >= len(best_path) and time.time() < deadline - 10:
            best_path_ref2 = [best_path]
            _dfs_extend(list(path), set(path), deadline, best_path_ref2)
            if len(best_path_ref2[0]) > len(best_path):
                best_path = best_path_ref2[0]
    
    return best_path
