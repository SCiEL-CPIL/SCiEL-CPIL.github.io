import time
import random

N = 9
V = 1 << N  # 512 vertices

# Precompute neighbors
NEIGHBORS = [[] for _ in range(V)]
for v in range(V):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def _generate_snake(start, rng, deadline, initial_path=None):
    """
    Generates a snake extending from the end of initial_path.
    Uses a greedy heuristic with lookahead (maximizing future options).
    """
    if initial_path is None:
        path = [start]
        used = {start}
        forbidden = set()
    else:
        path = list(initial_path)
        used = set(path)
        forbidden = set()
        # Build forbidden set: neighbors of v_i for i=0 to len(path)-3
        # Because for current end v_k, prev is v_{k-1}.
        # Forbidden should contain neighbors of v_0 ... v_{k-2}.
        # So indices 0 to k-2.
        # len(path) is k+1.
        # So indices 0 to (k+1)-3 = k-2.
        if len(path) >= 3:
            for i in range(len(path) - 2):
                for n in NEIGHBORS[path[i]]:
                    forbidden.add(n)
    
    while time.time() < deadline:
        if len(path) < 2:
            cur = path[-1]
            prev = None
        else:
            cur = path[-1]
            prev = path[-2]
            
        candidates = []
        for u in NEIGHBORS[cur]:
            if u in used:
                continue
            if u in forbidden:
                continue
            candidates.append(u)
            
        if not candidates:
            break
            
        best_cand = None
        max_score = -1
        
        if len(candidates) == 1:
            best_cand = candidates[0]
        else:
            for u in candidates:
                score = 0
                for w in NEIGHBORS[u]:
                    if w in used:
                        continue
                    if w in forbidden:
                        continue
                    if prev is not None:
                        if (w ^ prev).bit_count() == 1:
                            continue
                    score += 1
                if score > max_score:
                    max_score = score
                    best_cand = u
                    
        path.append(best_cand)
        used.add(best_cand)
        
        if prev is not None:
            for n in NEIGHBORS[prev]:
                forbidden.add(n)
                
    return path

def _extend_both_ends(snake, rng, deadline):
    if len(snake) < 2:
        return snake
        
    time_left = deadline - time.time()
    if time_left < 2.0:
        return snake
        
    # Extend end
    end_deadline = time.time() + time_left * 0.4
    snake_end = _generate_snake(snake[-1], rng, end_deadline, initial_path=snake)
    
    time_left = deadline - time.time()
    if time_left < 2.0:
        return max(snake, snake_end, key=len)
        
    # Extend start
    snake_rev = snake_end[::-1]
    start_deadline = time.time() + time_left * 0.4
    snake_start = _generate_snake(snake_rev[-1], rng, start_deadline, initial_path=snake_rev)
    
    best = max(snake, snake_end, snake_start[::-1], key=len)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    while time.time() < deadline - 5.0:
        start = rng.randint(0, V - 1)
        current_path = _generate_snake(start, rng, deadline)
        
        if len(current_path) > best_len and time.time() < deadline - 10.0:
            current_path = _extend_both_ends(current_path, rng, deadline)
            
        if len(current_path) > best_len:
            best_len = len(current_path)
            best_path = current_path
            
    if len(best_path) < 2:
        return [0, 1]
        
    # Verification
    used_set = set(best_path)
    if len(used_set) != len(best_path):
        return [0, 1]
        
    for i in range(len(best_path) - 1):
        if (best_path[i] ^ best_path[i + 1]).bit_count() != 1:
            return [0, 1]
            
    for i in range(len(best_path)):
        for j in range(i + 2, len(best_path)):
            if (best_path[i] ^ best_path[j]).bit_count() < 2:
                return [0, 1]
                
    return best_path
