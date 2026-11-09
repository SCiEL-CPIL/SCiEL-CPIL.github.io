import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def can_add(S, x):
        """Check if adding x to set S keeps it valid. S is a set."""
        if x in S:
            return True
        # Triples involving x:
        # Case 1: x is the smallest (a=x, b,c in S with b>c... no b<=c and b>x, c>=b)
        #   x < b <= c, x | (b+c)
        # Case 2: x is the middle (a < x <= c, a in S, a < x, c in S, c >= x)
        #   a < x <= c, a | (x+c)
        # Case 3: x is the largest (a < b <= x, a,b in S, a < b, b <= x)
        #   a < b <= x, a | (b+x)
        
        S_list = sorted(S)
        
        # Case 1: a=x, need b,c in S with x < b <= c and x | (b+c)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: b=x, need a,c in S with a < x <= c and a | (x+c)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: c=x, need a,b in S with a < b <= x and a | (b+x)
        for a in S_list:
            for b in S_list:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def can_add_optimized(S, x):
        """Optimized check using sorted list."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Case 1: a=x, b,c in S with x < b <= c, x | (b+c)
        # b+c must be a multiple of x, and b > x, c >= b
        # b+c ranges from 2*(x+1) to 2*max(S)
        # Multiples of x in that range
        for b in S_list:
            if b <= x:
                continue
            # c must be >= b, in S, and (b+c) % x == 0
            # c ≡ -b (mod x)
            target = (-b) % x
            if target == 0:
                target = x
            # Need c in S, c >= b, c ≡ target (mod x)
            # Just check all c >= b
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: b=x, a,c in S with a < x, c >= x, a | (x+c)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: c=x, a,b in S with a < b <= x, a | (b+x)
        for a in S_list:
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    # Start with upper tail [171, 256]
    best = set(range(171, 257))
    best_size = len(best)
    
    # Greedily add elements from 1 to 170
    for candidate in range(1, 171):
        if time.time() > deadline - 5:
            break
        if candidate in best:
            continue
        if can_add_optimized(best, candidate):
            best.add(candidate)
            best_size += 1
    
    # Try different addition orders to see if we can get a larger set
    # Order 1: Ascending (1 to 170)
    # Order 2: Descending (170 to 1)
    
    orders = [
        list(range(1, 171)),
        list(range(170, 0, -1))
    ]
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        S = set(range(171, 257))
        for candidate in order:
            if time.time() > deadline - 5:
                break
            if candidate in S:
                continue
            if can_add_optimized(S, candidate):
                S.add(candidate)
        
        if len(S) > best_size:
            best = S
            best_size = len(S)
            
    return sorted(best)
