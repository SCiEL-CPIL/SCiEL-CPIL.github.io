import time
from math import ceil

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # LCM of all divisors
    def lcm(a, b):
        return a * b // __import__('math').gcd(a, b)
    
    period = 1
    for d in divisors:
        period = lcm(period, d)
    
    best_span = float('inf')
    best_result = None
    best_L = 0
    
    for L in range(period):
        if time.time() > deadline:
            break
        
        # For each divisor, find the first multiple >= L
        # Then check distinctness, and if conflicts, adjust
        reps = []
        for d in divisors:
            # First multiple of d >= L
            k = (L + d - 1) // d
            reps.append(k * d)
        
        # Check if all distinct
        if len(set(reps)) == n:
            span = max(reps) - L
            if span < best_span:
                best_span = span
                best_result = reps[:]
                best_L = L
        else:
            # Need to resolve conflicts - try pushing conflicting ones up
            # This is more complex; let's use a greedy approach
            # Actually, let's try a different approach: for each divisor,
            # the multiple can be any multiple >= L. We want min max(reps) - L
            # with all reps distinct.
            # Greedy: sort divisors by their first multiple, assign in order,
            # if conflict, bump to next multiple
            # But this greedy may not be optimal. Let's try a more careful approach.
            # For now, let's also try the case where we bump the later ones.
            # Actually, let me just do a proper search for each L.
            pass
    
    # The above simple approach may miss cases where bumping gives a better result
    # Let me redo with a proper approach for each L
    
    best_span = float('inf')
    best_result = None
    
    for L in range(period):
        if time.time() > deadline:
            break
        
        # For each divisor d_i, the candidates are multiples of d_i >= L
        # We want to pick one for each, all distinct, minimizing max(reps) - L
        # 
        # Key insight: the optimal R is determined by the maximum rep.
        # Let's binary search on R? Or just try R = max of first multiples, then adjust.
        
        # First, compute the "natural" first multiple for each
        firsts = []
        for d in divisors:
            k = (L + d - 1) // d
            firsts.append(k * d)
        
        # The minimum possible max is at least max(firsts)
        # Try R = max(firsts), check if we can assign all distinct values <= R
        # For each divisor, the available multiples in [L, R] are:
        #   ceil(L/d)*d, ceil(L/d)*d + d, ..., up to R
        
        # Let's try a matching/greedy approach
        # Sort divisors by their number of available multiples (ascending) - most constrained first
        # Actually, let's try: for R starting from max(firsts), check feasibility
        
        R_candidate = max(firsts)
        # Check if we can assign distinct multiples
        # For each divisor d_i, multiples in [L, R_candidate]:
        #   k from ceil(L/d_i) to floor(R_candidate/d_i)
        
        # Use a simple greedy: process divisors in order of fewest options
        options = []
        for i, d in enumerate(divisors):
            k_start = (L + d - 1) // d
            k_end = R_candidate // d
            if k_end >= k_start:
                opts = list(range(k_start * d, k_end * d + 1, d))
            else:
                opts = []
            options.append((len(opts), i, opts))
        
        options.sort(key=lambda x: x[0])
        
        used = set()
        assignment = [None] * n
        feasible = True
        for _, i, opts in options:
            found = False
            for val in opts:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    found = True
                    break
            if not found:
                feasible = False
                break
        
        if feasible:
            span = max(assignment) - L
            if span < best_span:
                best_span = span
                best_result = assignment[:]
    
    if best_result is None:
        # Fallback: just use the divisors themselves (they are distinct)
        best_result = divisors[:]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]
