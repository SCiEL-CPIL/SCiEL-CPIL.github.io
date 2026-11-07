import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 128
    L = set(range(N//2 + 1, N + 1)) # 65..128
    small_candidates = list(range(1, N//2 + 1)) # 1..64
    
    best_A = set()
    best_size = 0
    
    def is_valid_addition(x, A):
        # Check constraint 1: x + y <= 64 for all y in A
        for y in A:
            if x + y > 64:
                return False
        
        # Check constraint 2: No distinct y, z in A such that
        # x+y in A, x+z in A, and (y+z in A or y+z == x)
        A_list = list(A)
        n = len(A_list)
        for i in range(n):
            y = A_list[i]
            if (x + y) not in A:
                continue
            for j in range(i + 1, n):
                z = A_list[j]
                if (x + z) in A:
                    if (y + z) in A or (y + z) == x:
                        return False
        return True

    orders = [
        sorted(small_candidates),
        sorted(small_candidates, reverse=True),
    ]
    
    for _ in range(100):
        random.shuffle(small_candidates)
        orders.append(small_candidates[:])
        
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        current_A = set()
        for x in order:
            if is_valid_addition(x, current_A):
                current_A.add(x)
                
        if len(current_A) > best_size:
            best_size = len(current_A)
            best_A = set(current_A)
            
    result_set = L.union(best_A)
    result = [int(x) for x in result_set]
    
    return result
