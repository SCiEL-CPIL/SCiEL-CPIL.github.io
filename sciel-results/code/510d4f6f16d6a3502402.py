import time

def construct():
    best_set = []
    start_time = time.time()
    limit = 55.0 # Leave 5s buffer
    
    # Deterministic Greedy Algorithm
    # We iterate x from 1 to 4096.
    # We maintain a set of all valid triple sums so far.
    # A triple is a multiset of 3 elements from the current set.
    # Unordered triples with repetition: (a, b, c) with a <= b <= c.
    # Sums are a+b+c.
    
    current_set = []
    sums_set = set()
    
    for x in range(1, 4097):
        # Check if x can be added
        # We need to check all new triples involving x.
        # Triples are of the form:
        # 1. (x, x, x) -> 3x
        # 2. (x, x, y) where y in current_set, y != x (but y < x since we iterate increasing, so y < x always) -> 2x + y
        # 3. (x, y, z) where y, z in current_set, y <= z, y != x, z != x.
        #    Since x is the largest element added so far (we iterate 1..N), any y, z in current_set are < x.
        #    So we just need to check all pairs (y, z) from current_set.
        
        new_sums = []
        
        # 1. (x, x, x)
        new_sums.append(3 * x)
        
        # 2. (x, x, y) for y in current_set
        for y in current_set:
            new_sums.append(2 * x + y)
            
        # 3. (x, y, z) for y, z in current_set with y <= z
        k = len(current_set)
        for i in range(k):
            y = current_set[i]
            for j in range(i, k):
                z = current_set[j]
                new_sums.append(x + y + z)
                
        # Check for internal conflicts in new_sums
        # If any sum in new_sums is already in sums_set, or if new_sums has duplicates, x is invalid.
        valid = True
        seen_in_new = set()
        for s_val in new_sums:
            if s_val in sums_set or s_val in seen_in_new:
                valid = False
                break
            seen_in_new.add(s_val)
            
        if valid:
            current_set.append(x)
            sums_set.update(new_sums)
            
    return current_set
