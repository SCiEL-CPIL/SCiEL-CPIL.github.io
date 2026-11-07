import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 128
    base_set = set(range(N//2 + 1, N + 1)) 
    small_numbers = list(range(1, N//2 + 1)) 
    
    best_set = set(base_set)
    best_size = len(best_set)
    
    def can_add(x, S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                s1 = x + a
                s2 = x + b
                s3 = a + b
                if s1 in S and s2 in S:
                    if s3 in S or s3 == x:
                        return False
        return True

    def run_greedy(candidates_order):
        current_set = set(base_set)
        for x in candidates_order:
            if x in current_set:
                continue
            if can_add(x, current_set):
                current_set.add(x)
        return current_set

    # Try fixed orders
    orders = [
        small_numbers[::-1], # Descending
        small_numbers[:],    # Ascending
    ]
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        s = run_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random restarts
    while time.time() < deadline - 5:
        candidates = small_numbers[:]
        random.shuffle(candidates)
        s = run_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    result = [int(x) for x in best_set]
    return result
