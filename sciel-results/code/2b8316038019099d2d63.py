import time
import random

N = 9
MAX_V = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initial Deterministic Greedy
    best_path = _greedy_deterministic()
    best_len = len(best_path)
    
    # 2. Iterated Local Search
    while time.time() < deadline:
        # Strategy A: Random Start + Randomized Greedy
        if random.random() < 0.5:
            path = _randomized_greedy_walk()
        else:
            # Strategy B: Truncate Best + Randomized Greedy
            path = _truncate_and_extend(best_path)
            
        if path is not None and len(path) > best_len:
            best_len = len(path)
            best_path = path
            
            # If we found a significantly better path, try to extend it a bit more
            # by doing a few more truncation/extension cycles
            for _ in range(5):
                if time.time() > deadline:
                    break
                ext_path = _truncate_and_extend(best_path)
                if ext_path is not None and len(ext_path) > best_len:
                    best_len = len(ext_path)
                    best_path = ext_path
                else:
                    break
                    
    return best_path

def _greedy_deterministic():
    path = [0]
    used = {0}
    forbidden = set()
    
    while True:
        cur = path[-1]
        next_v = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or v in forbidden:
                continue
            next_v = v
            break
        
        if next_v is None:
            break
        
        path.append(next_v)
        used.add(next_v)
        
        prev_v = cur
        for b in range(N):
            n = prev_v ^ (1 << b)
            if n != next_v:
                forbidden.add(n)
                
    return path

def _randomized_greedy_walk():
    start_v = random.randint(0, MAX_V - 1)
    path = [start_v]
    used = {start_v}
    forbidden = set()
    
    while True:
        cur = path[-1]
        valid_neighbors = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in forbidden:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        next_v = random.choice(valid_neighbors)
        path.append(next_v)
        used.add(next_v)
        
        prev_v = cur
        for b in range(N):
            n = prev_v ^ (1 << b)
            if n != next_v:
                forbidden.add(n)
                
    return path

def _truncate_and_extend(best_path):
    if not best_path:
        return None
        
    # Truncate to a random point in the last 50 vertices
    min_len = max(1, len(best_path) - 50)
    cut_point = random.randint(min_len, len(best_path) - 1)
    
    # Truncate
    path = best_path[:cut_point]
    
    # Rebuild used and forbidden
    used = set(path)
    forbidden = set()
    for i in range(len(path) - 1):
        v_i = path[i]
        for b in range(N):
            n = v_i ^ (1 << b)
            forbidden.add(n)
            
    # Randomized Greedy Extension
    while True:
        cur = path[-1]
        valid_neighbors = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and v not in forbidden:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        next_v = random.choice(valid_neighbors)
        path.append(next_v)
        used.add(next_v)
        
        prev_v = cur
        for b in range(N):
            n = prev_v ^ (1 << b)
            if n != next_v:
                forbidden.add(n)
                
    return path
