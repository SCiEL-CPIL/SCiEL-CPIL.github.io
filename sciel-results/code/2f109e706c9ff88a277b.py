import time

def construct():
    n = 256
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    def is_valid_with(S, x):
        """Check if adding x to S maintains the property."""
        if x in S:
            return True
        
        lst = sorted(S)
        m = len(lst)
        
        # 1. x is a: x < b <= c in S
        # For each b > x in S, check if x divides (b+c) for any c >= b in S
        for i in range(m):
            b = lst[i]
            if b <= x:
                continue
            for j in range(i, m):
                c = lst[j]
                if (b + c) % x == 0:
                    return False
        
        # 2. x is b: a < x <= c in S
        # For each a < x in S, check if a divides (x+c) for any c >= x in S
        for a in lst:
            if a >= x:
                continue
            for c in lst:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # 3. x is c: a < b < x in S
        # For each a < b < x in S, check if a divides (b+x)
        for i in range(m):
            a = lst[i]
            if a >= x:
                continue
            for j in range(i + 1, m):
                b = lst[j]
                if b >= x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def check_full_validity(S):
        """Verify the entire set S is valid."""
        lst = sorted(S)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i + 1, m):
                b = lst[j]
                for k in range(j, m):
                    c = lst[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Start with the safe upper set: [171, 256]
    # Condition: 3*min > 2*max => 3*171 = 513 > 512 = 2*256 ✓
    S = set(range(171, n + 1))
    
    # Greedy add smaller numbers from high to low
    for x in range(170, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_with(S, x):
            S.add(x)
    
    # Local search: try adding any missing number
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S:
                continue
            if is_valid_with(S, x):
                S.add(x)
                improved = True
    
    # Swap-based local search: try removing one element and adding another
    # to see if we can increase cardinality or maintain it with better structure
    best_S = set(S)
    best_size = len(S)
    
    for _ in range(20):  # Limited iterations for swaps
        if time.time() > deadline:
            break
        
        found_swap = False
        for remove_x in sorted(S):
            if time.time() > deadline:
                break
            
            S_without = S - {remove_x}
            
            for add_x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if add_x in S_without:
                    continue
                if add_x == remove_x:
                    continue
                
                # Check if we can add add_x to S_without
                if is_valid_with(S_without, add_x):
                    new_S = S_without | {add_x}
                    if len(new_S) > best_size:
                        best_S = new_S
                        best_size = len(new_S)
                        S = new_S
                        found_swap = True
                        break
                    elif len(new_S) == best_size:
                        # Same size, but maybe this opens up more additions later
                        # Try it if it allows further additions
                        old_S = set(S)
                        S = new_S
                        # Try to add more
                        extra_added = False
                        for y in range(1, n + 1):
                            if time.time() > deadline:
                                break
                            if y in S:
                                continue
                            if is_valid_with(S, y):
                                S.add(y)
                                extra_added = True
                                break
                        if extra_added:
                            best_S = set(S)
                            best_size = len(S)
                            found_swap = True
                        else:
                            S = old_S
        
        if not found_swap:
            break
    
    # Final verification
    S = best_S
    if not check_full_validity(S):
        # Fallback to the safe set
        S = set(range(171, n + 1))
    
    return list(S)
