import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 128
    in_set = [False] * (N+1)
    # Start with odds
    for i in range(1, N+1, 2):
        in_set[i] = True
    
    current_elements = [i for i in range(1, N+1) if in_set[i]]
    best_size = len(current_elements)
    best_set = set(current_elements)
    
    def check_conflict_with_new(cand, elements):
        n = len(elements)
        for i in range(n):
            a = elements[i]
            for j in range(i+1, n):
                b = elements[j]
                s_ab = a + b
                if s_ab > N:
                    continue
                if not (in_set[s_ab] or s_ab == cand):
                    continue
                
                s_ca = cand + a
                if s_ca > N:
                    continue
                if not in_set[s_ca]:
                    continue
                
                s_cb = cand + b
                if s_cb > N:
                    continue
                if not in_set[s_cb]:
                    continue
                
                return True
        return False

    # Initial Greedy
    for cand in range(1, N+1):
        if in_set[cand]:
            continue
        if not check_conflict_with_new(cand, current_elements):
            in_set[cand] = True
            current_elements.append(cand)
            if len(current_elements) > best_size:
                best_size = len(current_elements)
                best_set = set(current_elements)
                
    # Local Search
    # We will run for the remaining time
    while time.time() < deadline:
        # Randomly remove 1 element
        if not current_elements:
            break
        idx = random.randint(0, len(current_elements)-1)
        rem_elem = current_elements[idx]
        in_set[rem_elem] = False
        current_elements.pop(idx)
        
        # Try to add all
        for cand in range(1, N+1):
            if in_set[cand]:
                continue
            if not check_conflict_with_new(cand, current_elements):
                in_set[cand] = True
                current_elements.append(cand)
                if len(current_elements) > best_size:
                    best_size = len(current_elements)
                    best_set = set(current_elements)
        
        # Shuffle
        random.shuffle(current_elements)
        
        # Safety break
        if time.time() > deadline - 2:
            break

    return list(best_set)
