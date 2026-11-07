<<<<<<< SEARCH
    # Search for best pair (a, b) with sigma(a) == sigma(b) and |a/b - 3/2| minimized
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group (set of numbers with same sigma), for each b in the set,
    # check if a = (3b + k)/2 is in the set for small k.
    # We want to minimize |k| / (2b), so small |k| and large b.
    
    best_error = Fraction(1, 1)
    best_pair = None
    
    # Strategy: for each group, iterate b in descending order (larger b = smaller error)
    # For each b, check k = 1, -1, 3, -3, 5, -5, ... (odd k only since 2a-3b must be odd
    # when a,b are integers... wait, 2a is even, 3b is odd if b is odd, even if b is even.
    # 2a - 3b: if b is even, 3b is even, so 2a-3b is even, k is even.
    # If b is odd, 3b is odd, so 2a-3b is odd, k is odd.
    # So k can be any integer. But a = (3b+k)/2 must be integer, so 3b+k must be even.
    # If b is even: 3b is even, so k must be even.
    # If b is odd: 3b is odd, so k must be odd.
    
    # For efficiency, for each b, check a few small |k| values.
    # The error is |k|/(2b). To get a good result, we want small |k| and large b.
    
    # Let's iterate groups by size (larger groups more likely to have close pairs)
    # But first, let's just do a straightforward search with time limit.
    
    import numpy as np
    
    # Sort groups by size descending, process larger groups first (more likely to have good pairs)
    sorted_groups = sorted(filtered_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for s, group_set in sorted_groups:
        if time.time() > deadline - 5.0:
            break
        
        # Convert to sorted list for potential binary search, but we'll use set for O(1) lookup
        # For each b in the group (iterate in some order), check nearby a values
        
        # To be efficient, iterate b values and check a = (3b + k)/2 for small k
        # We want to check k values that make a integer and a in range [1, N]
        
        # Let's sample: check all b values but with limited k checks
        # For each b, check k in [-5, -3, -1, 1, 3, 5] (or appropriate parity)
        
        # Actually, let's be smarter. For each b, the best a is near 1.5*b.
        # Check a = floor(1.5*b) and ceil(1.5*b), and a few neighbors.
        
        # Convert set to list for iteration
        group_list = list(group_set)
        
        for b in group_list:
            if time.time() > deadline - 2.0:
                break
            
            # a should be close to 1.5 * b
            # a = (3b + k) / 2 for small k
            # Check k values with correct parity
            if b % 2 == 0:
                # b even, need k even
                k_values = [0, 2, -2, 4, -4, 6, -6, 8, -8]
            else:
                # b odd, need k odd
                k_values = [1, -1, 3, -3, 5, -5, 7, -7, 9, -9]
            
            for k in k_values:
                if k == 0:
                    continue  # invalid (error = 0)
                
                num_3b_plus_k = 3 * b + k
                if num_3b_plus_k <= 0 or num_3b_plus_k % 2 != 0:
                    continue
                
                a = num_3b_plus_k // 2
                if a < 1 or a > N or a == b:
                    continue
                
                if a in group_set:
                    # Valid pair!
                    err = Fraction(abs(k), 2 * b)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
    
    # If we found a very good pair (error < 1/1000), we can stop early
    # Otherwise, continue searching
    
    if best_pair is None:
        # Fallback: find any valid pair
        for s, group_set in filtered_groups.items():
            if len(group_set) >= 2:
                arr = sorted(group_set)
                a, b = arr[0], arr[1]
                if a != b:
                    best_pair = (a, b)
                    break
        if best_pair is None:
            best_pair = (11, 6)
    
    a, b = best_pair
    
    # Verify
    if int(sigma[a]) != int(sigma[b]):
        best_pair = (11, 6)
        a, b = best_pair
    
    return [int(a), int(b)]
=======
    # Search for best pair (a, b) with sigma(a) == sigma(b) and |a/b - 3/2| minimized
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group (set of numbers with same sigma), for each b in the set,
    # check if a = (3b + k)/2 is in the set for small k.
    # We want to minimize |k| / (2b), so small |k| and large b.
    
    best_error = Fraction(1, 1)
    best_pair = None
    
    import numpy as np
    
    # Sort groups by size descending, process larger groups first (more likely to have good pairs)
    sorted_groups = sorted(filtered_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for s, group_set in sorted_groups:
        if time.time() > deadline - 5.0:
            break
        
        # Convert set to sorted list, iterate b in descending order (larger b gives smaller error)
        group_list = sorted(group_set, reverse=True)
        
        for b in group_list:
            if time.time() > deadline - 2.0:
                break
            
            # a should be close to 1.5 * b
            # a = (3b + k) / 2 for small k
            # Check k values with correct parity: if b even, k even; if b odd, k odd
            # Expand k range to find tighter approximations
            max_k = 100
            if b % 2 == 0:
                k_values = list(range(2, max_k + 1, 2)) + list(range(-2, -max_k - 1, -2))
            else:
                k_values = list(range(1, max_k + 1, 2)) + list(range(-1, -max_k - 1, -2))
            
            found_in_b = False
            for k in k_values:
                num_3b_plus_k = 3 * b + k
                if num_3b_plus_k <= 0 or num_3b_plus_k % 2 != 0:
                    continue
                
                a = num_3b_plus_k // 2
                if a < 1 or a > N or a == b:
                    continue
                
                if a in group_set:
                    # Valid pair!
                    err = Fraction(abs(k), 2 * b)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
                        found_in_b = True
                        # Early termination if error is very small
                        if best_error <= Fraction(1, 10000):
                            break
            
            # If we found a very good pair, we can stop searching this group
            if best_error <= Fraction(1, 10000):
                break
    
    # If we found a very good pair (error < 1/1000), we can stop early
    # Otherwise, continue searching
    
    if best_pair is None:
        # Fallback: find any valid pair
        for s, group_set in filtered_groups.items():
            if len(group_set) >= 2:
                arr = sorted(group_set)
                a, b = arr[0], arr[1]
                if a != b:
                    best_pair = (a, b)
                    break
        if best_pair is None:
            best_pair = (11, 6)
    
    a, b = best_pair
    
    # Verify
    if int(sigma[a]) != int(sigma[b]):
        best_pair = (11, 6)
        a, b = best_pair
    
    return [int(a), int(b)]
>>>>>>> REPLACE
