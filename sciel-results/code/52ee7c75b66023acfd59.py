import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    # Precompute: for each divisor, store its value
    # We'll search over window widths W in increasing order.
    # For each W, we check all possible window starts L modulo LCM.
    # Key insight: the pattern of multiples repeats every LCM, so we only
    # need to check L in [1, LCM] for each W.
    
    for W in range(1, 200):
        if time.time() - start_time > deadline:
            # Fallback to a valid solution
            return [int(d) for d in divisors]
        
        for L in range(1, LCM + 1):
            if time.time() - start_time > deadline:
                return [int(d) for d in divisors]
            
            hi = L + W
            options = []
            valid = True
            for d in divisors:
                # First multiple of d >= L
                k = (L + d - 1) // d
                first = k * d
                if first > hi:
                    valid = False
                    break
                opts = []
                x = first
                while x <= hi:
                    opts.append(x)
                    x += d
                options.append(opts)
            
            if not valid:
                continue
            
            if any(len(o) == 0 for o in options):
                continue
            
            # Order by fewest options first (most constrained)
            order = sorted(range(n), key=lambda i: len(options[i]))
            
            # Use a bitmask for used values; values are in [L, L+W], so offset by L
            # Max W is small, so we can use a list of booleans or a set
            # For speed with small W, use a set of integers
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = order[idx]
                for val in options[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        
                        # Forward checking
                        feasible = True
                        for j in range(idx + 1, n):
                            k_idx = order[j]
                            if not any(v not in used for v in options[k_idx]):
                                feasible = False
                                break
                        
                        if feasible and backtrack(idx + 1):
                            return True
                        
                        used.remove(val)
                        assignment[i] = None
            
            if backtrack(0):
                if len(set(assignment)) == n:
                    if all(0 < x <= 1000000 for x in assignment):
                        if all(assignment[i] % divisors[i] == 0 for i in range(n)):
                            return [int(x) for x in assignment]
    
    # Fallback
    return [int(d) for d in divisors]
