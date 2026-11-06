import random
import time

def construct():
    start_time = time.time()
    time_limit = 55.0
    n = 4096
    best_set = []
    best_size = 0
    
    def evaluate_set(order):
        current_set = []
        current_sums = set()
        
        for x in order:
            new_sums = set()
            valid = True
            
            s = 3 * x
            if s in current_sums:
                valid = False
            else:
                new_sums.add(s)
                
            if valid:
                for y in current_set:
                    s1 = 2 * x + y
                    if s1 in current_sums or s1 in new_sums:
                        valid = False
                        break
                    new_sums.add(s1)
                    
                    s2 = x + 2 * y
                    if s2 in current_sums or s2 in new_sums:
                        valid = False
                        break
                    new_sums.add(s2)
                        
            if valid:
                cs = current_set
                for i in range(len(cs)):
                    y = cs[i]
                    for j in range(i + 1, len(cs)):
                        z = cs[j]
                        s = x + y + z
                        if s in current_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
                        
            if valid:
                current_set.append(x)
                current_sums.update(new_sums)
                
        return current_set

    # 1. Smallest First
    s_set = evaluate_set(range(1, n + 1))
    if len(s_set) > best_size:
        best_size = len(s_set)
        best_set = s_set
        
    # 2. Largest First
    if time.time() - start_time < time_limit:
        l_set = evaluate_set(range(n, 0, -1))
        if len(l_set) > best_size:
            best_size = len(l_set)
            best_set = l_set

    # 3. Random Restarts
    candidates = list(range(1, n + 1))
    
    while time.time() - start_time < time_limit:
        random.shuffle(candidates)
        r_set = evaluate_set(candidates)
        if len(r_set) > best_size:
            best_size = len(r_set)
            best_set = r_set
            if best_size >= 40:
                break

    return best_set
