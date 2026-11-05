import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    def check_and_update(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            # Verify validity
            if is_valid(current_set):
                best_set = list(current_set)
                best_size = len(current_set)
                
    def is_valid(S):
        # Bitset DP
        reachable = 1  # bit 0 is set
        for x in S:
            reachable |= (reachable << x)
            if (reachable >> 200) & 1:
                return False
        return True

    # 1. Deterministic Greedies
    # Large to Small
    s = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # Small to Large
    s = []
    reachable = 1
    for x in range(1, 101):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    check_and_update(s)
    
    # 2. Randomized Greedy
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        s = []
        reachable = 1
        for x in order:
            new_reach = reachable | (reachable << x)
            if not ((new_reach >> 200) & 1):
                s.append(x)
                reachable = new_reach
        check_and_update(s)
        
    return best_set
