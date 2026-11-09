import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    # LCM of all divisors
    import math
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
    # lcm = 30030
    
    best_val = float('inf')
    best_assign = None
    
    # For each window size W, try all starting positions L in [1, lcm]
    # and check if we can assign distinct multiples
    # W must be at least max(divisors) - 1 = 38 for feasibility
    
    for W in range(38, 300):
        if time.time() > deadline:
            break
        for L in range(1, lcm + 1):
            if time.time() > deadline:
                break
            if L + W > MAX_V:
                continue
            
            # For each divisor, find all multiples in [L, L+W]
            candidates = []
            for d in divisors:
                # First multiple >= L
                first = ((L + d - 1) // d) * d
                mults = []
                m = first
                while m <= L + W and m <= MAX_V:
                    mults.append(m)
                    m += d
                if not mults:
                    break
                candidates.append(mults)
            else:
                # All divisors have at least one candidate
                # Use backtracking to find distinct assignment
                assign = [None] * n
                used = set()
                
                def backtrack(idx):
                    if time.time() > deadline:
                        return False
                    if idx == n:
                        return True
                    for m in candidates[idx]:
                        if m not in used:
                            used.add(m)
                            assign[idx] = m
                            if backtrack(idx + 1):
                                return True
                            used.remove(m)
                            assign[idx] = None
                    return False
                
                if backtrack(0):
                    val = max(assign) - min(assign)
                    if val < best_val:
                        best_val = val
                        best_assign = assign[:]
                    # If we found a valid assignment, we can stop searching
                    # for this W since any other L with same W gives same or worse
                    # Actually no, we want minimum W, so once we find any valid
                    # for this W, we can break out of L loop
                    break
    
    if best_assign is None:
        # Fallback: just use the divisors themselves (which is valid but not optimal)
        return divisors[:]
    
    return [int(x) for x in best_assign]
