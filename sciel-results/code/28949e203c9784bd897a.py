import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    n = len(divisors)

    def next_multiple(M, d):
        return ((M + d - 1) // d) * d

    best_span = float('inf')
    best_solution = None

    # Strategy 1: Check critical M values (multiples of divisors)
    # For each such M, compute smallest multiple of each d >= M
    # The span is max(candidates) - min(candidates)
    # Since M is a multiple of some divisor, min(candidates) = M
    
    # We only need to check M values where at least one divisor has M as a multiple
    # and the resulting candidates are all distinct and <= max_val
    
    # Build sorted list of critical points (multiples of any divisor)
    # But limit to reasonable range since optimal span is small
    
    # First, do a quick scan over a bounded range
    # The optimal span is likely < 100, so we can bound our search
    
    scan_limit = 200000
    step = 1  # check every value initially, but we can be smarter
    
    # Actually, let's be smarter: only check M that are multiples of at least one divisor
    # Precompute which M values are multiples of at least one divisor up to scan_limit
    # But that's still many. Let's just iterate and check.
    
    # More efficient: iterate M from 6 to scan_limit, but skip M that aren't multiples of any divisor
    # Actually, the proof says min(x) must be a multiple of some divisor.
    # So we only need to check M that are multiples of at least one divisor.
    
    # Let's generate critical points efficiently
    critical = set()
    for d in divisors:
        for m in range(d, min(scan_limit, max_val) + 1, d):
            critical.add(m)
    
    critical = sorted(critical)
    
    for M in critical:
        if time.time() > deadline:
            break
        
        if M > max_val:
            break
        
        candidates = []
        valid = True
        for d in divisors:
            x = next_multiple(M, d)
            if x > max_val:
                valid = False
                break
            candidates.append(x)
        
        if not valid:
            continue
        
        # Check distinctness
        if len(set(candidates)) != n:
            continue
        
        span = max(candidates) - min(candidates)
        
        if span < best_span:
            best_span = span
            best_solution = candidates[:]
            
            if best_span <= 9:
                # Can't do better than 9 (10 distinct ints need span >= 9)
                break

    # Strategy 2: If we haven't found a great solution, try targeted search
    # for very small spans by checking windows of size W from 9 upward
    
    if best_span > 20:
        # Try to find solutions with span <= 50 by checking all M in a dense range
        # around promising areas
        import random
        random.seed(42)
        
        for _ in range(50000):
            if time.time() > deadline:
                break
            
            # Try random M values that are multiples of divisors
            d = random.choice(divisors)
            M = random.randint(1, min(scan_limit, max_val) // d) * d
            M = max(M, 6)
            
            if M > max_val:
                continue
            
            candidates = []
            valid = True
            for d2 in divisors:
                x = next_multiple(M, d2)
                if x > max_val:
                    valid = False
                    break
                candidates.append(x)
            
            if not valid:
                continue
            
            if len(set(candidates)) != n:
                continue
            
            span = max(candidates) - min(candidates)
            
            if span < best_span:
                best_span = span
                best_solution = candidates[:]
                
                if best_span <= 9:
                    break

    # Final fallback
    if best_solution is None:
        best_solution = divisors[:]

    return [int(x) for x in best_solution]
