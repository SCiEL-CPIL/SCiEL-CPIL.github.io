import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 4096
    best_set = []
    
    def get_new_sums(current_set, x):
        sums = [3 * x]
        for y in current_set:
            sums.append(2 * x + y)
        for i in range(len(current_set)):
            y = current_set[i]
            for j in range(i, len(current_set)):
                z = current_set[j]
                sums.append(x + y + z)
        return sums

    def is_valid(current_set, sums_set, x):
        if 3 * x in sums_set:
            return False
        for y in current_set:
            if 2 * x + y in sums_set:
                return False
        for i in range(len(current_set)):
            y = current_set[i]
            for j in range(i, len(current_set)):
                z = current_set[j]
                if x + y + z in sums_set:
                    return False
        return True

    def greedy_init():
        s = []
        sums = set()
        for x in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if is_valid(s, sums, x):
                s.append(x)
                sums.update(get_new_sums(s, x))
        return s

    # Initial greedy to get a baseline
    greedy_set = greedy_init()
    best_set = greedy_set
    
    def backtrack(current_set, sums_set, next_min):
        nonlocal best_set, deadline, start_time
        
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
        
        # Try to extend
        for x in range(next_min, n + 1):
            if time.time() > deadline:
                return
            
            if is_valid(current_set, sums_set, x):
                new_sums = get_new_sums(current_set, x)
                current_set.append(x)
                sums_set.update(new_sums)
                
                backtrack(current_set, sums_set, x + 1)
                
                current_set.pop()
                for s in new_sums:
                    sums_set.remove(s)
                
        return

    backtrack([], set(), 1)
    
    return best_set
