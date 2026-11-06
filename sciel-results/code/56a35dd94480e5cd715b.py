import time
import random

def construct():
    n = 256
    # Start with a high-density candidate set
    # Numbers > 128 have fewer multiples in the sum range [2a+2, 512]
    # Specifically, for a > 170, no multiples exist.
    # For a in [129, 170], 3a is in range.
    # We start with [129, 256] and prune.
    
    S = list(range(129, 257))
    S_set = set(S)
    
    def is_valid(S_list, S_set):
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                # c must be >= b
                # Check if any c in S_list[j:] makes a | (b+c)
                # b+c = k*a => c = k*a - b
                # c >= b => k*a - b >= b => k*a >= 2b => k >= ceil(2b/a)
                # c <= 256 => k*a - b <= 256 => k*a <= 256+b => k <= floor((256+b)/a)
                # Also c must be in S_set.
                
                # Optimization: precompute multiples? 
                # Just iterate k.
                min_k = (2 * b + a - 1) // a
                max_k = (256 + b) // a
                for k in range(min_k, max_k + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        return False
        return True

    def count_conflicts(S_list, S_set):
        """Returns a dict of element -> conflict count"""
        conflicts = {x: 0 for x in S_list}
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                min_k = (2 * b + a - 1) // a
                max_k = (256 + b) // a
                for k in range(min_k, max_k + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        # Conflict found: (a, b, c)
                        # Increment counts for a, b, c
                        # Note: c might not be in S_list index-wise if we are careful, 
                        # but it is in S_set.
                        # We need to increment the count for the specific values.
                        conflicts[a] += 1
                        conflicts[b] += 1
                        if c in conflicts: # c is in S
                            conflicts[c] += 1
        return conflicts

    # Pruning Phase
    # Keep removing the element with the highest conflict count until valid
    max_prune_steps = 100
    for _ in range(max_prune_steps):
        if is_valid(S, S_set):
            break
        
        conflicts = count_conflicts(S, S_set)
        if not conflicts:
            break
            
        # Find element with max conflicts
        max_conf = -1
        max_elem = None
        for elem, cnt in conflicts.items():
            if cnt > max_conf:
                max_conf = cnt
                max_elem = elem
        
        if max_elem is None:
            break
            
        S.remove(max_elem)
        S_set.remove(max_elem)
        
    # Now S is valid.
    # Greedy Addition Phase
    # Try to add elements from [1, 256] not in S
    # Prioritize adding smaller numbers? Or larger?
    # Larger numbers are "safer" as divisors? No, smaller divisors have more multiples.
    # Actually, large numbers as 'a' are safe. Small numbers as 'a' are dangerous.
    # But we want to maximize size.
    # Let's try adding elements in descending order? Or random?
    # Let's try all elements from 1 to 256.
    
    candidates = [x for x in range(1, 257) if x not in S_set]
    
    # To improve performance, we can shuffle candidates or sort them.
    # Sorting by "difficulty" might help.
    # Let's just iterate 1..256.
    
    for x in candidates:
        # Check if x can be added
        ok = True
        
        # Case 1: x is 'a'
        # Need to check if there exist b, c in S such that x < b <= c and x | (b+c)
        # Iterate b in S where b > x
        for b in S:
            if b <= x:
                continue
            # Check if there exists c in S, c >= b, such that x | (b+c)
            # c = k*x - b
            min_k = (2 * b + x - 1) // x
            max_k = (256 + b) // x
            for k in range(min_k, max_k + 1):
                c = k * x - b
                if c >= b and c in S_set:
                    ok = False
                    break
            if not ok:
                break
        
        if not ok:
            continue
            
        # Case 2: x is 'b'
        # Need to check if there exist a, c in S such that a < x <= c and a | (x+c)
        # Iterate a in S where a < x
        for a in S:
            if a >= x:
                continue
            # Check if there exists c in S, c >= x, such that a | (x+c)
            # c = k*a - x
            min_k = (x + x + a - 1) // a # c >= x => k*a - x >= x => k*a >= 2x
            max_k = (256 + x) // a
            for k in range(min_k, max_k + 1):
                c = k * a - x
                if c >= x and c in S_set:
                    ok = False
                    break
            if not ok:
                break
        
        if not ok:
            continue
            
        # Case 3: x is 'c'
        # Need to check if there exist a, b in S such that a < b <= x and a | (b+x)
        # Iterate b in S where b <= x
        for b in S:
            if b > x:
                continue
            # Check if there exists a in S, a < b, such that a | (b+x)
            # (b+x) % a == 0
            # Iterate a in S where a < b
            for a in S:
                if a >= b:
                    continue
                if (b + x) % a == 0:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            S.append(x)
            S_set.add(x)
            # Re-sort S to maintain order for potential future optimizations?
            # The check logic uses set membership and explicit loops, so order doesn't strictly matter for correctness, 
            # but keeping it sorted might help if we switch to binary search later.
            # For now, O(N) insert is fine.
            S.sort()

    return S

# EVOLVE-BLOCK-END
