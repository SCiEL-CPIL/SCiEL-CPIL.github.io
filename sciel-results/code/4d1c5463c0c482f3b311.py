import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe

    # We want to minimize W = max - min
    # Try W from 9 to some limit
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        # Try different starting positions L
        # The multiples are periodic. The pattern of availability repeats with period LCM(d_i).
        # LCM is huge, but we can just scan L.
        # A good range for L: 1 to 500 is usually enough to find a solution if one exists for that W.
        # Why? Because the "gaps" in multiples are small.
        for L in range(1, 500):
            if time.time() > deadline:
                break
            
            # For this window [L, L+W], find available multiples for each divisor
            # available[i] is a list of multiples of divisors[i] in [L, L+W]
            available = []
            for d in divisors:
                # First multiple >= L
                first_k = (L + d - 1) // d
                multiples = []
                m = first_k * d
                while m <= L + W:
                    multiples.append(m)
                    m += d
                available.append(multiples)
            
            # Check if any divisor has no available multiple
            if any(len(a) == 0 for a in available):
                continue
            
            # Solve assignment problem via backtracking
            # Order divisors by number of available multiples (ascending)
            # We need to track which indices in 'divisors' we are solving for
            order = sorted(range(len(divisors)), key=lambda i: len(available[i]))
            
            result = [None] * len(divisors)
            used_values = set()
            
            def backtrack(idx):
                if idx == len(order):
                    return True
                i = order[idx]
                for val in available[i]:
                    if val not in used_values:
                        used_values.add(val)
                        result[i] = val
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(val)
                        result[i] = None
                return False
            
            if backtrack(0):
                # Found a solution
                return result
            
    # Fallback: return divisors (score 1.0)
    return divisors[:]
