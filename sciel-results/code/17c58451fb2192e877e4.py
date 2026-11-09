import time

def construct():
    start_time = time.time()
    limit = 55.0
    
    current_set = []
    sums_set = set()
    
    for x in range(1, 4097):
        if time.time() - start_time > limit:
            break
            
        # Check if x can be added
        # We need to check:
        # 1. Internal collisions among new sums
        # 2. Collisions with existing sums
        
        # Optimization:
        # The new sums are:
        # 3x
        # 2x + s for s in current_set
        # x + si + sj for si <= sj in current_set
        
        # We can just generate them and check.
        
        new_sums = [3*x]
        
        for s in current_set:
            new_sums.append(2*x + s)
            
        k = len(current_set)
        for i in range(k):
            si = current_set[i]
            for j in range(i, k):
                sj = current_set[j]
                new_sums.append(x + si + sj)
                
        # Check internal duplicates
        if len(new_sums) != len(set(new_sums)):
            continue
            
        # Check against existing
        valid = True
        for s_val in new_sums:
            if s_val in sums_set:
                valid = False
                break
                
        if valid:
            current_set.append(x)
            sums_set.update(new_sums)
            
    return current_set
