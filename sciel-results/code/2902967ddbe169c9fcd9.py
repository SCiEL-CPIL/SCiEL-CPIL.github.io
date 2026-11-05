import time

def construct():
    start_time = time.time()
    
    A = [1]
    sums = {3}
    pair_sums = {2}  # 1+1
    
    # We iterate through candidates from 2 to 4096
    for x in range(2, 4097):
        if time.time() - start_time > 55:
            break
            
        k = len(A)
        
        # 1. Check S1 (3x) against sums
        if 3 * x in sums:
            continue
            
        # 2. Check S1 (3x) against S3 (x + pair_sum)
        # 3x = x + p => 2x = p
        if 2 * x in pair_sums:
            continue
            
        # 3. Generate S2 (2x + y) and check against sums
        # Also check S2 against S3
        # S2 = {2x + y for y in A}
        # S3 = {x + p for p in pair_sums}
        
        # Check S2 vs Sums
        valid = True
        for y in A:
            s_val = 2 * x + y
            if s_val in sums:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check S2 vs S3
        # 2x + y = x + p => x + y = p
        # For each y in A, check if x+y is in pair_sums
        for y in A:
            if x + y in pair_sums:
                valid = False
                break
                
        if not valid:
            continue
            
        # 4. Generate S3 (x + p) and check against sums
        # We don't need to generate the set, just check membership
        for p in pair_sums:
            s_val = x + p
            if s_val in sums:
                valid = False
                break
                
        if not valid:
            continue
            
        # All checks passed. Add x.
        A.append(x)
        
        # Update sums
        # Add 3x
        sums.add(3 * x)
        # Add 2x + y
        for y in A[:-1]: # A[:-1] is the old A
            sums.add(2 * x + y)
        # Add x + p
        for p in pair_sums:
            sums.add(x + p)
            
        # Update pair_sums
        # New pairs are (x, x) and (x, y) for y in old A
        pair_sums.add(2 * x)
        for y in A[:-1]:
            pair_sums.add(x + y)
            
    return A
