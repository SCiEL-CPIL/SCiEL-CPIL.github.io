import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit out of 90s

    # Base solution: upper tail from 171 to 256
    # This is valid because for any a<b<=c in [171,256], a >= 171
    # b+c >= 171+172 = 343, and 2*171 = 342, so b+c > 2a always
    # Actually we need a NOT dividing b+c.
    # For a in [171,256], b+c is in [343, 512]
    # a divides b+c means b+c = k*a for some integer k
    # k*a >= 343 and k*a <= 512
    # For a >= 171, k=1 gives a in [343,512] impossible
    # k=2 gives 2a in [343,512], so a in [172,256]
    # So we need to check if b+c = 2a for any a<b<=c in the set
    # But wait, if a is in the set and b,c are in the set with a<b<=c
    # and b+c = 2a, then a divides b+c.
    # For the tail [171,256], can we have b+c=2a?
    # a >= 171, b > a, c >= b
    # b+c >= (a+1)+(a+1) = 2a+2 > 2a
    # So b+c > 2a always, meaning b+c != 2a
    # And b+c can't equal a (too small) or 3a (too large for most)
    # 3a >= 513 > 512 max sum, so only k=2 is possible
    # Since b+c > 2a, the tail [171,256] is valid.
    
    best_set = set(range(171, 257))
    best_size = len(best_set)
    
    # Try to improve by adding elements below 171
    # For an element x < 171 to be added, we need to check
    # that for all a < b <= c in the resulting set, a does not divide b+c
    
    # Greedy approach: try adding smaller elements one by one
    # Check if adding element x to current set S is valid
    
    def is_valid_with(S, x):
        """Check if adding x to S keeps the set valid."""
        new_S = S | {x}
        # Check all triples a<b<=c in new_S
        elems = sorted(new_S)
        for i, a in enumerate(elems):
            for j in range(i+1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if b + c % a == 0:
                        return False
        return True
    
    # More efficient: only check triples involving x
    def can_add(S, x):
        """Check if x can be added to S without violating constraints."""
        if x in S:
            return True
        
        # Cases:
        # 1. x is the smallest: x < a < b <= c, need x not divide a+b
        # 2. x is middle: a < x < b <= c, need a not divide x+b and a not divide x+c
        # 3. x is largest: a < b < x, need a not divide b+x
        # 4. x equals b or c: handled by b<=c
        
        for a in S:
            if a < x:
                # a could be the small element, x is b or c
                for b in S:
                    if b > a:
                        if b == x or (b > x):
                            # a < x <= b, check a divides x+b
                            if (x + b) % a == 0:
                                return False
                            if (b > x and (b + x) % a == 0):
                                return False
                        elif b < x:
                            # a < b < x, check a divides b+x
                            if (b + x) % a == 0:
                                return False
                # Also need to check if x is a, so x < a
                # That's covered below
            elif a > x:
                # x < a, x could be the small element
                for b in S:
                    if b > a:
                        # x < a < b, check x divides a+b
                        if (a + b) % x == 0:
                            return False
                    elif b == x:
                        continue
                    elif b < a and b > x:
                        # x < b < a, check x divides b+a
                        if (b + a) % x == 0:
                            return False
        return True
    
    # Simpler and more correct: just check all triples involving x
    def can_add_correct(S, x):
        if x in S:
            return True
        
        # Generate all triples (a, b, c) with a < b <= c in S ∪ {x}
        # that involve x
        for a in S:
            if a < x:
                # x is b or c
                for b in S:
                    if b < x:
                        # a < b < x: check a | (b+x)
                        if (b + x) % a == 0:
                            return False
                        # b < a < x is not possible since a < b
                    elif b == x:
                        continue
                    else:  # b > x
                        # a < x < b: check a | (x+b)
                        if (x + b) % a == 0:
                            return False
                        # Also check x | (a+b) since x is middle? No, x is not a
                        # Wait, we need to check if x divides a+b when x is middle
                        # That's a separate case
            elif a > x:
                # x < a: x is the small element
                for b in S:
                    if b > a:
                        # x < a < b: check x | (a+b)
                        if (a + b) % x == 0:
                            return False
                    elif b < a:
                        # b < a, and x < a
                        if b > x:
                            # x < b < a: check x | (b+a)
                            if (b + a) % x == 0:
                                return False
                        elif b < x:
                            # b < x < a: check a | (b+x)? No, a is largest
                            # Check x | (a+b)
                            if (a + b) % x == 0:
                                return False
        return True
    
    # Even simpler: just do the full check for small candidate sets
    # But that's O(n^3) which is too slow
    
    # Let's use a different approach: try specific patterns
    
    # Try adding elements in decreasing order from 170 down to 1
    S = set(range(171, 257))
    
    for x in range(170, 0, -1):
        if time.time() > deadline - 10:
            break
        
        # Quick check: if x divides any sum b+c where b,c in S and x < b <= c
        # Or if any a in S divides x+b or x+c where appropriate
        
        valid = True
        for a in S:
            if a < x:
                # x is b or c, a is smaller
                for b in S:
                    if b >= a and b != x:
                        s = x + b if x >= b else b + x
                        if s % a == 0:
                            valid = False
                            break
            if not valid:
                break
            if a > x:
                # x is smaller, a is b or c
                for b in S:
                    if b > a:
                        if (a + b) % x == 0:
                            valid = False
                            break
            if not valid:
                break
            if a == x:
                continue
        
        if valid:
            S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
    
    # Try some random perturbations
    import random
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        S_trial = set(best_set)
        # Remove 1-3 random elements
        for _ in range(random.randint(1, 3)):
            if S_trial:
                S_trial.remove(random.choice(list(S_trial)))
        
        # Try to add back
        for x in range(256, 0, -1):
            if x not in S_trial:
                # Quick validity check
                valid = True
                for a in S_trial:
                    if a < x:
                        for b in S_trial:
                            if b > a and (x + b) % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    elif a > x:
                        for b in S_trial:
                            if b > a and (a + b) % x == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    S_trial.add(x)
        
        if len(S_trial) > best_size:
            best_size = len(S_trial)
            best_set = set(S_trial)
    
    # Final validation
    def full_check(S):
        elems = sorted(S)
        for i, a in enumerate(elems):
            for j in range(i+1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    if full_check(best_set):
        return list(best_set)
    
    # Fallback to known valid solution
    return list(range(171, 257))
