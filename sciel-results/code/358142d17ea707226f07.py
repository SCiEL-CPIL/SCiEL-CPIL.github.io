import time

def construct():
    LCM = 5354228880
    vals = [LCM // i for i in range(1, 25)]
    
    best_size = 0
    best_set = []
    
    def get_subset_sums(elements):
        sums = {0}
        for v in elements:
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + v)
            sums = new_sums
        return sums

    def is_valid_addition(x, S, subset_sums):
        val_x = vals[x-1]
        
        # Check 1: 1/x not in sums of S
        if val_x in subset_sums:
            return False
            
        # Check 2: For each y in S, 1/y - 1/x not in sums of S \ {y}
        for y in S:
            val_y = vals[y-1]
            target = val_y - val_x
            if target < 0:
                continue
            
            S_without_y = [z for z in S if z != y]
            sums_without_y = get_subset_sums([vals[z-1] for z in S_without_y])
            
            if target in sums_without_y:
                return False
                
        return True

    # Greedy initialization
    S_greedy = []
    sums_greedy = {0}
    for x in range(24, 0, -1):
        if is_valid_addition(x, S_greedy, sums_greedy):
            S_greedy.append(x)
            val_x = vals[x-1]
            new_sums = set(sums_greedy)
            for s in sums_greedy:
                new_sums.add(s + val_x)
            sums_greedy = new_sums
            
    best_size = len(S_greedy)
    best_set = S_greedy[:]
    
    start_time = time.time()
    
    def dfs(idx, S, subset_sums, count):
        nonlocal best_size, best_set
        
        if time.time() - start_time > 50:
            return
            
        if count + (24 - idx) <= best_size:
            return
            
        if idx == 0:
            if count > best_size:
                best_size = count
                best_set = S[:]
            return
        
        x = idx
        
        # Try exclude
        dfs(idx - 1, S, subset_sums, count)
        
        # Try include
        if is_valid_addition(x, S, subset_sums):
            new_S = S + [x]
            val_x = vals[x-1]
            new_sums = set(subset_sums)
            for s in subset_sums:
                new_sums.add(s + val_x)
            dfs(idx - 1, new_S, new_sums, count + 1)

    dfs(24, [], {0}, 0)
    
    return best_set
