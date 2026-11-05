import time
import random

N = 256
DEADLINE = 85.0

# Precompute 4-APs
aps_containing = [[] for _ in range(N + 1)]

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        for x in ap:
            aps_containing[x].append(ap)

def is_valid_addition(A_bool, y):
    for ap in aps_containing[y]:
        if all(A_bool[x] for x in ap if x != y):
            return False
    return True

def count_3aps_involving(A_bool, x):
    count = 0
    max_d_mid = min(x - 1, N - x)
    for d in range(1, max_d_mid + 1):
        if A_bool[x - d] and A_bool[x + d]:
            count += 1
            
    max_d_first = (N - x) // 2
    for d in range(1, max_d_first + 1):
        if A_bool[x + d] and A_bool[x + 2 * d]:
            count += 1
            
    max_d_last = x // 2
    for d in range(1, max_d_last + 1):
        if A_bool[x - 2 * d] and A_bool[x - d]:
            count += 1
            
    return count

def total_3aps(A_bool):
    total = 0
    for b in range(1, N + 1):
        if A_bool[b]:
            max_d = min(b - 1, N - b)
            for d in range(1, max_d + 1):
                if A_bool[b - d] and A_bool[b + d]:
                    total += 1
    return total

def construct():
    start_time = time.time()
    best_set = set()
    best_score = -1
    
    # 20 restarts, 20 iterations each
    for seed in range(20):
        random.seed(seed)
        if time.time() - start_time > DEADLINE:
            break
            
        A_bool = [False] * (N + 1)
        A_set = set()
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(A_bool, x):
                A_bool[x] = True
                A_set.add(x)
        
        current_score = total_3aps(A_bool)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(A_set)
            
        improved = True
        for _ in range(20):
            if not improved: break
            if time.time() - start_time > DEADLINE: break
            
            best_swap = None
            best_delta = 0
            
            xs = list(A_set)
            ys = [y for y in range(1, N + 1) if not A_bool[y]]
            
            random.shuffle(xs)
            random.shuffle(ys)
            
            for x in xs:
                if time.time() - start_time > DEADLINE: break
                
                loss_x = count_3aps_involving(A_bool, x)
                A_bool[x] = False
                
                for y in ys:
                    if time.time() - start_time > DEADLINE: break
                    
                    if not is_valid_addition(A_bool, y):
                        continue
                    
                    gain_y = count_3aps_involving(A_bool, y)
                    delta = gain_y - loss_x
                    if delta > best_delta:
                        best_delta = delta
                        best_swap = (x, y)
                
                A_bool[x] = True
                
            if best_swap:
                x, y = best_swap
                A_bool[x] = False
                A_set.remove(x)
                A_bool[y] = True
                A_set.add(y)
                current_score += best_delta
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(A_set)

    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
