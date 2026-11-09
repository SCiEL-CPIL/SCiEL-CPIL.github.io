def construct():
    def is_no_zero_base11(n):
        if n == 0:
            return False
        while n > 0:
            if n % 11 == 0:
                return False
            n //= 11
        return True

    def can_add(S, x):
        # Check if adding x to S creates a 3-AP
        # x is not in S
        for a in S:
            # Case 1: x is the middle term. a, x, c
            # 2x = a + c => c = 2x - a
            c = 2 * x - a
            if c != x and c in S:
                return False
            
            # Case 2: x is the last term. a, b, x
            # a + x = 2b => b = (a + x) / 2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b != x and b != a and b in S:
                    return False
            
            # Case 3: x is the first term. x, b, a
            # x + a = 2b => b = (x + a) / 2
            # This is the same as Case 2 with a and x swapped?
            # s = x + a. b = s/2.
            # If b in S, then x, b, a is an AP.
            # We already checked if b in S.
            # So Case 2 covers both "x is last" and "x is first" regarding the midpoint.
            # Wait, if x is first: x, b, a. Midpoint b.
            # Condition: b in S.
            # My check `if b != x and b != a and b in S` covers this.
            
        return True

    # 1. Greedy Construction on S0
    S = set()
    for x in range(1, 257):
        if not is_no_zero_base11(x):
            continue
        if can_add(S, x):
            S.add(x)
    
    # 2. Local Search
    # Try to swap larger elements in S with smaller elements not in S
    improved = True
    while improved:
        improved = False
        # Sort S to iterate
        S_list = list(S)
        for x in range(1, 257):
            if x in S:
                continue
            # Try to find a y in S such that y > x and swap is valid
            # We prefer to swap out the largest possible y? Or any y?
            # Swapping out any y > x increases sum.
            # To maximize gain, we might want to swap out the largest y?
            # No, the gain is 1/x - 1/y. This is maximized when y is large.
            # So iterate y from largest to smallest.
            for y in sorted(S_list, reverse=True):
                if y <= x:
                    break
                # Check if (S - {y}) + {x} is valid
                # Since S is valid, S - {y} is valid.
                # We just need to check if x forms an AP with any pair in S - {y}.
                valid = True
                S_minus_y = S - {y}
                if can_add(S_minus_y, x):
                    S.remove(y)
                    S.add(x)
                    improved = True
                    S_list = list(S) # Update list
                    break
        
    return list(S)
