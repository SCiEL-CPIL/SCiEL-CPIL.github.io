import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    prescribed = {1, 2, 5, 11}
    max_val = 284
    
    def get_sums(s):
        sums = set()
        s_list = sorted(s)
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        return sums

    # Initial greedy
    current_set = list(prescribed)
    current_set.sort()
    sums = get_sums(current_set)
    
    # Greedy add
    for x in range(1, max_val + 1):
        if x in current_set:
            continue
        conflict = False
        for y in current_set:
            if x + y in sums:
                conflict = True
                break
        if not conflict and 2 * x not in sums:
            current_set.append(x)
            current_set.sort()
            sums = get_sums(current_set)
            
    best_set = current_set[:]
    best_size = len(best_set)
    
    stuck_count = 0
    
    while time.time() < deadline:
        if len(current_set) >= 20:
            break
            
        # Try to add
        added = False
        for _ in range(100):
            x = random.randint(1, max_val)
            if x in current_set:
                continue
            conflict = False
            for y in current_set:
                if x + y in sums:
                    conflict = True
                    break
            if not conflict and 2 * x not in sums:
                current_set.append(x)
                current_set.sort()
                sums = get_sums(current_set)
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                stuck_count = 0
                break
        
        if added:
            continue
            
        # Try swap
        swapped = False
        for _ in range(100):
            if len(current_set) == 0: 
                break
            idx = random.randint(0, len(current_set) - 1)
            x = current_set[idx]
            y = random.randint(1, max_val)
            if y == x or y in current_set:
                continue
            
            # Efficient validity check for swap
            s_rem = [u for u in current_set if u != x]
            sums_rem = set()
            for i in range(len(s_rem)):
                for j in range(i, len(s_rem)):
                    sums_rem.add(s_rem[i] + s_rem[j])
            
            valid = True
            if 2 * y in sums_rem:
                valid = False
            else:
                for u in s_rem:
                    if (y + u) in sums_rem:
                        valid = False
                        break
            
            if valid:
                current_set[idx] = y
                current_set.sort()
                sums = get_sums(current_set)
                swapped = True
                stuck_count = 0
                break
        
        if not swapped and not added:
            stuck_count += 1
            if stuck_count > 100:
                # Reset to best known state to escape local optimum
                current_set = best_set[:]
                sums = get_sums(current_set)
                stuck_count = 0

    return best_set
