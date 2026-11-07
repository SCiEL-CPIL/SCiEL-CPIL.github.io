import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for d in divisors:
        period = lcm(period, d)
    # period = 30030
    
    best_span = float('inf')
    best_result = None
    
    # Binary search on span S
    # Lower bound: 9 (10 distinct integers need span >= 9)
    # Upper bound: period (worst case)
    lo, hi = 9, period
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Check if span = mid is feasible
        # For a window [L, L+mid], each divisor d_i must have at least one multiple in [L, L+mid]
        # And we need to assign distinct multiples.
        
        # Key insight: For each divisor d_i, the set of valid L values (mod period) where
        # there exists a multiple of d_i in [L, L+mid] is:
        # L must satisfy: there exists k such that L <= k*d_i <= L+mid
        # i.e., L in [k*d_i - mid, k*d_i] for some k
        # So valid L mod period is the union of intervals [k*d_i - mid, k*d_i] mod period
        
        # Build the set of valid L residues for each divisor
        # Then find intersection across all divisors
        
        # For efficiency, work with the union of intervals on [0, period)
        # For divisor d, multiples are at 0, d, 2d, ..., (period/d - 1)*d
        # Valid L: for some multiple m, L in [m - mid, m] (mod period)
        
        valid_sets = []
        for d in divisors:
            # Build union of intervals [m - mid, m] mod period for all multiples m of d
            # Represent as list of (start, end) intervals on [0, period)
            intervals = []
            num_mults = period // d
            for k in range(num_mults):
                m = k * d
                start = m - mid
                end = m
                # Normalize to [0, period)
                if start >= 0:
                    if end < period:
                        intervals.append((start, end))
                    else:
                        intervals.append((start, period - 1))
                        intervals.append((0, end % period))
                else:
                    # start < 0, wraps around
                    # [start, end] = [start, 0) union [0, end]
                    if end < period:
                        intervals.append((0, end))
                    # And [start, 0) wraps to [period + start, period - 1]
                    intervals.append((period + start, period - 1))
            
            # Merge overlapping intervals
            if not intervals:
                valid_sets.append([])
                continue
            
            intervals.sort()
            merged = [list(intervals[0])]
            for s, e in intervals[1:]:
                if s <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            
            valid_sets.append(merged)
        
        # Now find intersection of all valid_sets
        # Start with the first set, intersect with each subsequent
        # Represent current intersection as list of intervals
        
        if not valid_sets[0]:
            continue
        
        current = [list(iv) for iv in valid_sets[0]]
        
        for i in range(1, n):
            if not current:
                break
            if not valid_sets[i]:
                current = []
                break
            
            # Intersect two sets of intervals
            new_current = []
            j1 = j2 = 0
            while j1 < len(current) and j2 < len(valid_sets[i]):
                s1, e1 = current[j1]
                s2, e2 = valid_sets[i][j2]
                lo_i = max(s1, s2)
                hi_i = min(e1, e2)
                if lo_i <= hi_i:
                    new_current.append([lo_i, hi_i])
                if e1 < e2:
                    j1 += 1
                else:
                    j2 += 1
            current = new_current
        
        if current:
            # Span mid is feasible
            hi = mid - 1
            best_span = mid
        else:
            lo = mid + 1
    
    if best_span == float('inf') or best_span > period:
        best_span = period
    
    # Now find the actual assignment for span = best_span
    # We need to find L such that we can assign distinct multiples
    # Search over L in [0, period)
    
    S = best_span
    found = False
    
    for L in range(period):
        if time.time() > deadline:
            break
        
        # For each divisor, find multiples in [L, L+S]
        # Try to assign distinct values using backtracking
        
        # Get options for each divisor
        options = []
        for i, d in enumerate(divisors):
            k_start = (L + d - 1) // d
            k_end = (L + S) // d
            if k_end >= k_start:
                opts = list(range(k_start * d, k_end * d + 1, d))
            else:
                opts = []
            options.append((len(opts), i, opts))
        
        # Check all have at least one option
        if any(len(opts) == 0 for _, _, opts in options):
            continue
        
        # Sort by fewest options (most constrained first)
        options.sort(key=lambda x: x[0])
        
        # Backtracking
        used = set()
        assignment = [None] * n
        
        def backtrack(idx):
            if idx == len(options):
                return True
            _, i, opts = options[idx]
            for val in opts:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assignment[i] = None
            return False
        
        if backtrack(0):
            found = True
            break
    
    if found:
        best_result = [int(x) for x in assignment]
    else:
        # Fallback: just use the divisors themselves
        best_result = [int(x) for x in divisors]
    
    return best_result
