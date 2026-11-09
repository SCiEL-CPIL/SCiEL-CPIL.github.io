import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    
    # 1. Greedy Heuristic to get initial best
    greedy_set = set()
    greedy_sum = 0.0
    for x in range(1, N + 1):
        valid = True
        for a in greedy_set:
            # Check if x, a, b or a, x, b or a, b, x forms AP
            # 1. x is middle: a, x, b => b = 2x - a
            b = 2 * x - a
            if b in greedy_set and b != a and b != x:
                valid = False
                break
            # 2. x is end: a, b, x => b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in greedy_set and b != a and b != x:
                    valid = False
                    break
            # 3. x is start: x, a, b => b = 2a - x
            b = 2 * a - x
            if b in greedy_set and b != a and b != x:
                valid = False
                break
        if valid:
            greedy_set.add(x)
            greedy_sum += 1.0 / x
            
    best_sum = greedy_sum
    best_set = set(greedy_set)
    
    # Precompute suffix sums for B&B bound
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i+1] + 1.0/i
        
    time_out = False
    
    def is_conflict(x, s):
        for a in s:
            b = 2 * x - a
            if b in s and b != a and b != x:
                return True
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in s and b != a and b != x:
                    return True
            b = 2 * a - x
            if b in s and b != a and b != x:
                return True
        return False

    def dfs(idx, s, curr_sum):
        nonlocal best_sum, best_set, time_out
        
        if time.time() > deadline:
            time_out = True
            return
            
        if curr_sum + suffix_sum[idx] <= best_sum:
            return
            
        if idx > N:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_set = set(s)
            return
        
        # Branch 1: Include idx
        if not is_conflict(idx, s):
            s.add(idx)
            dfs(idx + 1, s, curr_sum + 1.0/idx)
            s.remove(idx)
            
        if time_out:
            return
            
        # Branch 2: Exclude idx
        dfs(idx + 1, s, curr_sum)

    # Start B&B
    # We can start from 1, but we already have a good baseline.
    # The B&B will try to find a set with sum > best_sum.
    dfs(1, set(), 0.0)
    
    if not best_set:
        return [1]
        
    return list(best_set)
