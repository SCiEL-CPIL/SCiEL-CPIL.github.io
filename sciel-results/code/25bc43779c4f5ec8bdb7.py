import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    def can_add(x, S_list, S_set):
        K = []
        # Local variable for speed
        s_set = S_set
        for a in S_list:
            if (x + a) in s_set:
                K.append(a)
        
        if len(K) < 2:
            return True
            
        k_len = len(K)
        for i in range(k_len):
            ki = K[i]
            for j in range(i + 1, k_len):
                if (ki + K[j]) in s_set:
                    return False
        return True

    def greedy_search(order):
        S_set = set()
        S_list = []
        for x in order:
            if can_add(x, S_list, S_set):
                S_set.add(x)
                S_list.append(x)
        return S_set

    # Pre-generate a few static orders? No, random is better.
    
    # To improve quality, we can try to bias the random order.
    # E.g., start with numbers > 64, then < 64?
    # Or just uniform.
    
    while time.time() < deadline:
        order = list(range(1, 129))
        random.shuffle(order)
        
        S = greedy_search(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return list(best_set)
