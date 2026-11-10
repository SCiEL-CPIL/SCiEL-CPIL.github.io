def construct():
    n = 12
    
    # Step 1: Generate greedy solution
    subset_sums = {0}
    result = []
    used = set()
    for _ in range(n):
        v = 1
        while True:
            if v not in used:
                new_sums = set()
                valid = True
                for s in subset_sums:
                    ns = s + v
                    if ns in subset_sums or ns in new_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                if valid:
                    subset_sums.update(new_sums)
                    result.append(v)
                    used.add(v)
                    break
            v += 1
    
    # Step 2: Local search to minimize max element
    # We try to reduce elements from largest index to smallest.
    # Note: We must maintain distinctness of the integers themselves.
    
    current = result[:]
    
    def is_valid(arr):
        sums = {0}
        for v in arr:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    # Optimization: Precompute sums for prefixes?
    # For simplicity, just run is_valid.
    
    # Try to reduce each element
    for i in range(n - 1, -1, -1):
        # Try to reduce current[i]
        # Lower bound: must be > current[i-1] (if i>0) and > 0
        # Also must be distinct from others.
        low = 1
        if i > 0:
            low = current[i-1] + 1
        # We want to find the smallest valid value >= low
        # that is < current[i]
        
        found = False
        for new_val in range(low, current[i]):
            # Check distinctness with other elements
            if new_val in current[:i] or new_val in current[i+1:]:
                continue
            
            test_arr = current[:]
            test_arr[i] = new_val
            
            if is_valid(test_arr):
                current = test_arr
                found = True
                # Since we are scanning from low upwards, the first valid one is the smallest
                # But we want to minimize the max, so reducing current[i] is good.
                # We can break because we found a smaller valid value.
                # However, maybe an even smaller one exists?
                # We are iterating from low to current[i]-1.
                # So the first one found is the smallest valid value in that range.
                # So we can break.
                break
        
        if not found:
            # Try to see if we can reduce it further by checking if current[i] itself is minimal?
            # We already tried all values below current[i].
            # So current[i] is the minimal valid value given the other elements fixed.
            pass

    return current
