import random
import time

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    # Greedy initialization
    chosen = []
    counts_dict = {}
    for x in range(1, 1001):
        valid = True
        for y in chosen:
            s = x + y
            c = counts_dict.get(s, 0) + 1
            if c > 2:
                valid = False
                break
        if valid:
            for y in chosen:
                s = x + y
                counts_dict[s] = counts_dict.get(s, 0) + 1
            counts_dict[2*x] = counts_dict.get(2*x, 0) + 1
            chosen.append(x)
            
    # Initialize state for local search
    in_set = [False] * 1001
    for x in chosen:
        in_set[x] = True
    S = set(chosen)
    S_list = list(S)
    
    # Precompute counts for all sums
    counts = [0] * 2001
    for i in range(len(chosen)):
        for j in range(i, len(chosen)):
            s = chosen[i] + chosen[j]
            counts[s] += 1
            
    current_size = len(S)
    best_size = current_size
    best_S = set(S)
    
    def can_add(x):
        if counts[2*x] + 1 > 2:
            return False
        for y in S_list:
            s = x + y
            if counts[s] + 1 > 2:
                return False
        return True

    def do_add(x):
        nonlocal S_list, counts, current_size
        in_set[x] = True
        S.add(x)
        S_list.append(x)
        counts[2*x] += 1
        for y in S_list:
            if y == x:
                continue
            s = x + y
            counts[s] += 1
        current_size += 1

    def do_remove(x):
        nonlocal S_list, counts, current_size
        in_set[x] = False
        S.remove(x)
        S_list.remove(x)
        counts[2*x] -= 1
        for y in S_list:
            s = x + y
            counts[s] -= 1
        current_size -= 1

    while time.time() - start_time < timeout:
        # Try to add a random element
        if current_size < 1000:
            candidates = [x for x in range(1, 1001) if not in_set[x]]
            if candidates:
                x = random.choice(candidates)
                if can_add(x):
                    do_add(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                    continue
        
        # Try a swap: remove one, add one
        if S_list:
            y = random.choice(S_list)
            do_remove(y)
            
            candidates = [x for x in range(1, 1001) if not in_set[x]]
            found = False
            # Try a few random candidates
            for _ in range(20):
                if not candidates:
                    break
                x = random.choice(candidates)
                if can_add(x):
                    do_add(x)
                    found = True
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                    break
            
            if not found:
                # Revert removal
                do_add(y)
                
    return list(best_S)
