import math
import time
import itertools

def _score_of(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    sums = set()
    diffs = set()
    u_list = list(U)
    n = len(u_list)
    for i in range(n):
        for j in range(n):
            sums.add(u_list[i] + u_list[j])
            diffs.add(u_list[i] - u_list[j])
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return 1.0 # or 1 + 0
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom

def _greedy_construct(pool, k, deadline):
    current = {0}
    remaining = set(pool) - {0}
    
    for _ in range(k - 1):
        if time.time() > deadline:
            break
        best_x = None
        best_score = -1
        for x in remaining:
            candidate = current | {x}
            s = _score_of(candidate)
            if s > best_score:
                best_score = s
                best_x = x
        if best_x is not None:
            current.add(best_x)
            remaining.remove(best_x)
        else:
            break
            
    return list(current)

def _local_search(U, pool, max_steps, deadline):
    current = set(U)
    best_score = _score_of(current)
    current_list = list(current)
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        improved = False
        # Try adding a new element
        for x in pool:
            if x not in current:
                candidate = current | {x}
                s = _score_of(candidate)
                if s > best_score:
                    best_score = s
                    current = candidate
                    improved = True
                    break
        if improved:
            continue
            
        # Try removing an element (if size > 2)
        if len(current) > 2:
            for x in list(current):
                if x == 0: continue
                candidate = current - {x}
                s = _score_of(candidate)
                if s > best_score:
                    best_score = s
                    current = candidate
                    improved = True
                    break
        if improved:
            continue

        # Try swapping
        for x in list(current):
            if x == 0: continue
            for y in pool:
                if y not in current:
                    candidate = (current - {x}) | {y}
                    s = _score_of(candidate)
                    if s > best_score:
                        best_score = s
                        current = candidate
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
            
    return list(current)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with the seed
    best_set = [0, 1, 3]
    best_score = _score_of(best_set)
    
    # Try greedy construction for various k
    pool = list(range(0, 101))
    
    for k in range(3, 11):
        if time.time() > deadline - 10.0:
            break
        # Greedy
        g_set = _greedy_construct(pool, k, deadline - 5.0)
        g_score = _score_of(g_set)
        if g_score > best_score:
            best_score = g_score
            best_set = g_set
            
        # Local search from greedy result
        if time.time() > deadline - 5.0:
            break
        l_set = _local_search(g_set, pool, 100, deadline - 2.0)
        l_score = _score_of(l_set)
        if l_score > best_score:
            best_score = l_score
            best_set = l_set
            
    # Also try specific sequences
    sequences = [
        list(range(0, 10)),
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 8, 27, 64, 125],
        [0, 2, 5, 10, 17, 26], # Quadratic
    ]
    for seq in sequences:
        s = _score_of(seq)
        if s > best_score:
            best_score = s
            best_set = seq
            
    return best_set
