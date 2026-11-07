import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 70  # safety margin
    
    best_span = float('inf')
    best_result = None
    
    # Try natural assignments for L in a range
    for L in range(1, 100001):
        if time.time() > deadline:
            break
        reps = []
        valid = True
        for d in divisors:
            m = ((L + d - 1) // d) * d
            if m > max_val:
                valid = False
                break
            reps.append(m)
        
        if not valid:
            continue
        
        # Check distinctness
        if len(set(reps)) == n:
            span = max(reps) - min(reps)
            if span < best_span:
                best_span = span
                best_result = reps[:]
                if best_span == 1:  # minimal possible for distinct ints
                    break
        else:
            # Try greedy assignment with bumps
            used = set()
            greedy_reps = []
            for i, d in enumerate(divisors):
                m = ((L + d - 1) // d) * d
                # Find smallest multiple >= L that is not in used
                while m in used or m > max_val:
                    m += d
                if m > max_val:
                    break
                used.add(m)
                greedy_reps.append(m)
            
            if len(greedy_reps) == n:
                span = max(greedy_reps) - min(greedy_reps)
                if span < best_span:
                    best_span = span
                    best_result = greedy_reps[:]
                    if best_span == 1:
                        break
    
    # If no result found, fallback to divisors themselves (shouldn't happen)
    if best_result is None:
        best_result = divisors[:]
    
    # Ensure types are correct
    return [int(x) for x in best_result]
