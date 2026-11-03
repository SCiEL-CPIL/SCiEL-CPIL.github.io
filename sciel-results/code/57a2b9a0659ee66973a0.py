import time
import random
import math

def _score(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    u_list = list(U)
    n = len(u_list)
    for i in range(n):
        for j in range(n):
            sumset.add(u_list[i] + u_list[j])
            diffset.add(u_list[i] - u_list[j])
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1.0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _local_search(start_set, deadline):
    current = list(start_set)
    current_score = _score(current)
    best_local = list(current)
    best_local_score = current_score
    
    while time.time() < deadline:
        improved = False
        max_c = max(current) if current else 0
        
        # 1. Try adding
        # Limit search space to reasonable range
        upper_bound = 2 * max_c + 20
        candidates = list(range(0, upper_bound + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in current:
                continue
            new_set = current + [x]
            s = _score(new_set)
            if s > current_score:
                current = new_set
                current_score = s
                improved = True
                if s > best_local_score:
                    best_local_score = s
                    best_local = list(current)
                break
        
        # 2. Try removing
        if not improved:
            for x in current:
                if x == 0: continue
                new_set = [y for y in current if y != x]
                s = _score(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    improved = True
                    if s > best_local_score:
                        best_local_score = s
                        best_local = list(current)
                    break
        
        # 3. Try moving
        if not improved:
            for i, x in enumerate(current):
                if x == 0: continue
                # Random move
                y = random.randint(0, 2 * max_c + 10)
                if y in current:
                    continue
                new_set = list(current)
                new_set[i] = y
                s = _score(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    improved = True
                    if s > best_local_score:
                        best_local_score = s
                        best_local = list(current)
                    break
            
            if not improved:
                 # Small step move
                 for i, x in enumerate(current):
                     if x == 0: continue
                     for delta in [-1, 1, -2, 2]:
                         y = x + delta
                         if y < 0 or y in current: continue
                         new_set = list(current)
                         new_set[i] = y
                         s = _score(new_set)
                         if s > current_score:
                             current = new_set
                             current_score = s
                             improved = True
                             if s > best_local_score:
                                 best_local_score = s
                                 best_local = list(current)
                             break
                     if improved: break
        
        if not improved:
            # Stuck, break or randomize
            # If set is small, try a bigger jump
            if len(current) < 20:
                # Add a random large element
                y = random.randint(0, 5 * max_c + 50)
                if y not in current:
                    current = current + [y]
                    current_score = _score(current)
            else:
                break

    return best_local, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Multiple restarts
    # Different initial structures
    starts = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 6, 10],
        [0, 1, 4, 9],
        [0, 2, 3]
    ]
    
    # Add some random starts
    for _ in range(5):
        size = random.randint(3, 8)
        s = {0}
        max_val = 50
        while len(s) < size:
            s.add(random.randint(1, max_val))
        starts.append(list(s))
        
    for start in starts:
        if time.time() > deadline:
            break
        # Allow some time for each restart
        # Simple time allocation
        s_set, s_score = _local_search(start, deadline)
        if s_score > best_score:
            best_score = s_score
            best_set = s_set
            
    return best_set
