import time
import sys
import random

N = 9
TIME_LIMIT = 110.0  # Safety margin from 120s
start_time = time.time()
best_path = []
best_len = 0

# Precompute neighbors for each vertex
neighbors = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

def check_time():
    return time.time() - start_time > TIME_LIMIT

def is_valid_extension(v, path):
    """Check if v can be added to the end of path as an induced path extension."""
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def count_valid_extensions(v, path, used_mask):
    """Count how many valid extensions exist from vertex v given current path and used mask."""
    count = 0
    for next_v in neighbors[v]:
        if used_mask & (1 << next_v):
            continue
        # Check induced path constraint
        valid = True
        for p in path[:-1]:
            if (next_v ^ p).bit_count() == 1:
                valid = False
                break
        if valid:
            count += 1
    return count

def warnsdorff_sort(neighbor_list, path, used_mask):
    """Sort neighbors using Warnsdorff's rule: prefer vertices with fewer future extensions."""
    scored = []
    for v in neighbor_list:
        if used_mask & (1 << v):
            continue
        # Check induced path constraint
        valid = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                valid = False
                break
        if not valid:
            continue
        ext_count = count_valid_extensions(v, path, used_mask)
        scored.append((ext_count, v))
    scored.sort(key=lambda x: x[0])  # Prefer vertices with fewer extensions (more constrained)
    return [v for _, v in scored]

def dfs(v, path, used_mask, depth=0):
    global best_path, best_len
    
    if check_time():
        return
        
    current_len = len(path)
    if current_len > best_len:
        best_len = current_len
        best_path = path.copy()
        
    # Get valid neighbors sorted by Warnsdorff's rule
    valid_neighbors = warnsdorff_sort(neighbors[v], path, used_mask)
    
    for next_v in valid_neighbors:
        path.append(next_v)
        used_mask |= (1 << next_v)
        
        dfs(next_v, path, used_mask, depth + 1)
        
        path.pop()
        used_mask ^= (1 << next_v)
        
        if check_time():
            return

def local_search_extend(path, used_mask):
    """Try to extend the path using greedy local search with randomization."""
    global best_path, best_len
    path = list(path)
    used_mask = used_mask
    
    while not check_time():
        current = path[-1]
        valid_next = []
        for v in neighbors[current]:
            if used_mask & (1 << v):
                continue
            if is_valid_extension(v, path):
                valid_next.append(v)
        
        if not valid_next:
            break
        
        # Pick the next vertex with Warnsdorff's rule
        scored = []
        for v in valid_next:
            ext_count = count_valid_extensions(v, path, used_mask)
            scored.append((ext_count, v))
        scored.sort(key=lambda x: x[0])
        
        next_v = scored[0][1]
        path.append(next_v)
        used_mask |= (1 << next_v)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
    
    return path, used_mask

def construct():
    global start_time, best_path, best_len
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Symmetry breaking: Start at 0
    start_v = 0
    path = [start_v]
    used_mask = 1 << start_v
    
    sys.setrecursionlimit(10000)
    
    # Phase 1: DFS with Warnsdorff's heuristic
    dfs(start_v, path, used_mask)
    
    # Phase 2: Local search extension on the best path found
    if best_path and not check_time():
        local_search_extend(best_path, 0)
        # Reconstruct used_mask from best_path
        used_mask = 0
        for v in best_path:
            used_mask |= (1 << v)
        local_search_extend(best_path, used_mask)
    
    return best_path
