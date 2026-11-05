"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9

# EVOLVE-BLOCK-START
def _is_valid_candidate(v, path, used):
    """Check if v can be appended to path: not used, and distance >= 2 from all non-last vertices."""
    if v in used:
        return False
    # Check distance from all vertices except the last one (which is the current endpoint)
    for i in range(len(path) - 1):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True

def _greedy_extend(path, used, start_bit_order=None):
    """Greedy extension of a snake. Returns the final path."""
    if start_bit_order is None:
        start_bit_order = list(range(N))
    while True:
        cur = path[-1]
        nxt = None
        for b in start_bit_order:
            v = cur ^ (1 << b)
            if _is_valid_candidate(v, path, used):
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)

def _backtracking_search(path, used, max_depth=0):
    """Try to extend the path with backtracking. Returns the longest path found."""
    best_path = list(path)
    if max_depth > 0 and len(path) >= max_depth:
        return best_path
    
    cur = path[-1]
    candidates = []
    for b in range(N):
        v = cur ^ (1 << b)
        if _is_valid_candidate(v, path, used):
            candidates.append(v)
    
    if not candidates:
        return best_path
    
    # Try each candidate in random order for diversity
    random.shuffle(candidates)
    
    for v in candidates:
        path.append(v)
        used.add(v)
        
        # Recurse
        result = _backtracking_search(path, used, max_depth)
        if len(result) > len(best_path):
            best_path = result
        
        path.pop()
        used.remove(v)
    
    return best_path

def _backtracking_search_limited(path, used, time_limit, deadline):
    """Backtracking search with time limit. Returns the longest path found."""
    best_path = list(path)
    
    if time.time() > deadline:
        return best_path
    
    cur = path[-1]
    candidates = []
    for b in range(N):
        v = cur ^ (1 << b)
        if _is_valid_candidate(v, path, used):
            candidates.append(v)
    
    if not candidates:
        return best_path
    
    # Try candidates in random order
    random.shuffle(candidates)
    
    for v in candidates:
        if time.time() > deadline:
            return best_path
        
        path.append(v)
        used.add(v)
        
        # Recurse with depth limit to prevent excessive backtracking
        result = _backtracking_search_limited(path, used, time_limit, deadline)
        if len(result) > len(best_path):
            best_path = result
        
        path.pop()
        used.remove(v)
    
    return best_path

def construct():
    """Construct the longest snake in Q_9 using greedy + backtracking with multiple restarts."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_snake = []
    
    # Phase 1: Simple greedy from vertex 0
    path = [0]
    used = {0}
    snake = _greedy_extend(path, used)
    best_snake = snake
    
    # Phase 2: Try greedy from different starting vertices
    for start in range(1, 512):
        if time.time() > deadline:
            break
        
        path = [start]
        used = {start}
        snake = _greedy_extend(path, used)
        if len(snake) > len(best_snake):
            best_snake = snake
    
    # Phase 3: Backtracking search from the best greedy result
    if time.time() < deadline - 5.0:
        path = list(best_snake)
        used = set(best_snake)
        
        # Try to extend with backtracking
        extended = _backtracking_search_limited(path, used, 50, deadline)
        if len(extended) > len(best_snake):
            best_snake = extended
    
    # Phase 4: More greedy restarts with shuffled bit orders
    for trial in range(200):
        if time.time() > deadline:
            break
        
        start = random.randint(0, 511)
        bit_order = list(range(N))
        random.shuffle(bit_order)
        
        path = [start]
        used = {start}
        snake = _greedy_extend(path, used, bit_order)
        if len(snake) > len(best_snake):
            best_snake = snake
    
    # Phase 5: Final backtracking polish on the best snake
    if time.time() < deadline - 2.0:
        path = list(best_snake)
        used = set(best_snake)
        extended = _backtracking_search_limited(path, used, 30, deadline)
        if len(extended) > len(best_snake):
            best_snake = extended
    
    return best_snake
# EVOLVE-BLOCK-END
