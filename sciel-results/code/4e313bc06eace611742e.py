import time
import random

def construct():
    deadline = time.time() + 85  # 85s deadline with 5s safety margin
    n = 128
    
    # Precompute: for each pair (i,j), their sum i+j
    # A triple (a,b,c) with a<b<c is bad if a+b, a+c, b+c are all in the set
    
    def count_violations(S):
        """Count number of forbidden triples in set S."""
        S_set = set(S)
        violations = 0
        elems = sorted(S)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                for k in range(j+1, len(elems)):
                    a, b, c = elems[i], elems[j], elems[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        violations += 1
        return violations
    
    def is_valid(S):
        return count_violations(S) == 0
    
    # Start with the known valid set: {65, ..., 128}
    base_set = list(range(65, 129))
    best_set = base_set[:]
    best_size = len(best_set)
    
    # Try to greedily add elements from 1 to 64
    current = set(base_set)
    for x in range(1, 65):
        if time.time() > deadline - 30:
            break
        trial = current | {x}
        if is_valid(list(trial)):
            current = trial
            if len(current) > best_size:
                best_size = len(current)
                best_set = list(current)
    
    # Local search: try swapping elements
    # Represent set as a boolean array for speed
    def set_to_arr(S):
        arr = [False] * (n + 1)
        for x in S:
            arr[x] = True
        return arr
    
    def arr_to_set(arr):
        return [i for i in range(1, n+1) if arr[i]]
    
    def count_violations_fast(arr):
        """Fast violation count using array representation."""
        violations = 0
        elems = [i for i in range(1, n+1) if arr[i]]
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i+1, len(elems)):
                b = elems[j]
                ab = a + b
                if ab > n:
                    continue
                if not arr[ab]:
                    continue
                for k in range(j+1, len(elems)):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac <= n and bc <= n and arr[ac] and arr[bc]:
                        violations += 1
        return violations
    
    # Start local search from current best
    arr = set_to_arr(best_set)
    cur_violations = count_violations_fast(arr)
    
    # Simulated annealing style local search
    import math
    T = 1.0
    cooling = 0.999
    
    for iteration in range(200000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random element to toggle
        x = random.randint(1, n)
        if arr[x]:
            # Try removing x
            arr[x] = False
            new_violations = count_violations_fast(arr)
            delta_v = new_violations - cur_violations
            if delta_v < 0 or random.random() < math.exp(-delta_v / max(T, 1e-10)):
                cur_violations = new_violations
                if cur_violations == 0:
                    new_size = sum(arr)
                    if new_size > best_size:
                        best_size = new_size
                        best_set = arr_to_set(arr)
            else:
                arr[x] = True
        else:
            # Try adding x
            arr[x] = True
            new_violations = count_violations_fast(arr)
            delta_v = new_violations - cur_violations
            if delta_v < 0 or random.random() < math.exp(-delta_v / max(T, 1e-10)):
                cur_violations = new_violations
                if cur_violations == 0:
                    new_size = sum(arr)
                    if new_size > best_size:
                        best_size = new_size
                        best_set = arr_to_set(arr)
            else:
                arr[x] = False
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Final greedy pass: try to add any remaining elements
    final_set = set(best_set)
    for x in range(1, n+1):
        if x not in final_set and time.time() > deadline - 2:
            break
        trial = final_set | {x}
        if is_valid(list(trial)):
            final_set = trial
    
    if len(final_set) > best_size:
        best_set = list(final_set)
        best_size = len(best_set)
    
    # Ensure we return a valid list of Python ints
    result = sorted(best_set)
    # Verify validity one more time
    if not is_valid(result):
        # Fallback to known valid set
        result = list(range(65, 129))
    
    return result
