import math
import time
import random
import itertools

def compute_score(U):
    if not U or 0 not in U:
        return -1.0
    
    U = sorted(list(set(U)))
    n = len(U)
    max_u = max(U)
    
    # Compute difference set
    diff_set = set()
    sum_set = set()
    
    for a in U:
        for b in U:
            diff_set.add(a - b)
            sum_set.add(a + b)
            
    size_diff = len(diff_set)
    size_sum = len(sum_set)
    
    if size_sum == 0:
        return -1.0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        # Score is 1 + log(ratio)/log(...)
        # If ratio < 1, log is negative, score < 1.
        # We want to maximize, so prefer ratio > 1.
        pass
        
    denom = math.log2(2 * max_u + 1)
    if denom == 0:
        return -1.0
        
    score = 1.0 + math.log2(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    def local_search(U, max_iters=5000):
        U = sorted(list(set(U)))
        current_score = compute_score(U)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            n = len(U)
            
            # 1. Try adding elements
            # Limit search range to avoid huge numbers which hurt denominator
            # Heuristic: max_u should be proportional to n^2 or similar
            max_u = max(U)
            search_limit = max(20, 2 * max_u + 10, n * n * 2)
            
            candidates_add = []
            for x in range(1, search_limit):
                if x not in U:
                    candidates_add.append(x)
            
            random.shuffle(candidates_add)
            for x in candidates_add[:50]: # Limit number of adds to try per step
                new_U = U + [x]
                s = compute_score(new_U)
                if s > current_score + 1e-9:
                    U = sorted(new_U)
                    current_score = s
                    improved = True
                    break
            
            # 2. Try removing elements
            if improved:
                continue
                
            candidates_remove = [x for x in U if x != 0]
            random.shuffle(candidates_remove)
            for x in candidates_remove[:20]:
                new_U = [y for y in U if y != x]
                s = compute_score(new_U)
                if s > current_score + 1e-9:
                    U = sorted(new_U)
                    current_score = s
                    improved = True
                    break
            
            # 3. Try replacing elements
            if improved:
                continue
                
            # Replace small elements with larger ones? Or vice versa?
            # Usually keeping max small is good.
            # Try replacing a large element with a smaller one not in U
            if len(U) > 1:
                candidates_replace_from = [x for x in U if x != 0]
                random.shuffle(candidates_replace_from)
                for x in candidates_replace_from[:10]:
                    max_u = max(U)
                    # Try replacing x with something smaller?
                    # Or something larger to increase diff set?
                    # Let's try random candidates in [1, max_u+10]
                    for y in range(1, max_u + 10):
                        if y == x or y in U:
                            continue
                        new_U = [z if z != x else y for z in U]
                        s = compute_score(sorted(new_U))
                        if s > current_score + 1e-9:
                            U = sorted(new_U)
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved and iters % 100 == 0:
                 # Restart or jitter?
                 pass
        
        return U, current_score

    # Multi-start strategy
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if r == 0:
            init_U = [0, 1, 3]
        elif r == 1:
            init_U = [0, 1, 4, 6]
        elif r == 2:
            init_U = [0, 1, 4, 9, 11]
        else:
            # Random small set
            n = random.randint(3, 8)
            # Generate random increasing sequence
            max_val = random.randint(10, 50)
            init_U = sorted(random.sample(range(1, max_val), n-1))
            init_U.insert(0, 0)
            
        curr_U, curr_score = local_search(init_U)
        
        if curr_score > best_score:
            best_score = curr_score
            best_set = curr_U
            
    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    print(compute_score(res))
