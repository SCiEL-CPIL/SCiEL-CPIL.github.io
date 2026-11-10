"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Precompute: for each divisor, the list of multiples up to max_val
    # multiples[i] = list of multiples of divisors[i] up to max_val
    multiples = []
    for d in divisors:
        multiples.append(list(range(d, max_val + 1, d)))
    
    # For a given start S and window size W, try to assign distinct multiples
    # Greedy: for each divisor, pick the smallest multiple >= S that hasn't been used
    # and is <= S + W
    def try_assign(S, W):
        used = set()
        result = []
        for i in range(n):
            d = divisors[i]
            mults = multiples[i]
            # Find smallest multiple >= S and <= S+W not in used
            # Use binary search to find first multiple >= S
            lo, hi = 0, len(mults)
            while lo < hi:
                mid = (lo + hi) // 2
                if mults[mid] < S:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is index of first multiple >= S
            found = False
            for idx in range(lo, len(mults)):
                v = mults[idx]
                if v > S + W:
                    break
                if v not in used:
                    used.add(v)
                    result.append(v)
                    found = True
                    break
            if not found:
                return None
        return result
    
    # Strategy: enumerate window starts, try to fit in window W
    # For each start S, the best assignment uses smallest available multiples >= S
    # We want to minimize W = max(assigned) - S
    
    # Enumerate candidate starts: for each divisor i, candidate starts are
    # multiples of divisors[i] (since the min of the set must be a multiple of some divisor)
    # Actually, the start of the window doesn't have to be a multiple, but the min assigned value
    # must be a multiple of its divisor.
    
    # Better approach: for each possible "anchor" value (a multiple of some divisor),
    # try to build the tightest set containing that value.
    
    # Even better: enumerate all possible values that could be the minimum,
    # and for each, greedily assign the rest to minimize the maximum.
    
    # Candidate minimums: any multiple of any divisor
    # But that's too many. Instead, note that the minimum must be a multiple of
    # at least one divisor. Let's enumerate multiples of each divisor as potential minimums.
    
    # For efficiency, let's try a different approach:
    # For each divisor i, for each multiple m of divisors[i], treat m as the minimum,
    # and greedily assign the other divisors to the smallest available multiples >= m.
    # Track the best (max - m).
    
    best_span = float('inf')
    best_result = None
    deadline = time.time() + 75  # safety margin before 90s kill
    
    # Collect candidate minimums: multiples of each divisor
    # To limit candidates, only consider multiples that are "reasonable"
    # The span will be at least something, so we can bound the search
    
    # For each divisor, generate multiples as candidate minimums
    # But 1M / 6 ≈ 166666 candidates per divisor, times 10 = 1.6M, each requiring O(n log n)
    # That might be too slow. Let's be smarter.
    
    # Alternative: for each divisor i and each multiple m of divisors[i],
    # the greedy assignment of others is deterministic. We can compute it fast.
    
    # Let's precompute for each divisor the sorted list of multiples.
    # For a given minimum m (which must be a multiple of divisor j),
    # for each other divisor i, we need the smallest multiple of divisors[i] >= m
    # that is not equal to any other assigned value.
    
    # Since we're assigning in order and picking smallest available, conflicts are rare
    # for small windows. Let's just do it directly.
    
    # To keep it fast, iterate over candidate minimums but limit the range.
    # The optimal span is likely small (maybe < 1000). So for a candidate min m,
    # we only need to check multiples in [m, m + best_span).
    
    # Start with a decent upper bound: just use the divisors themselves
    initial = divisors[:]
    initial_span = max(initial) - min(initial)
    best_span = initial_span
    best_result = initial
    
    # Now try to improve by searching around each divisor value and nearby multiples
    # For each divisor d_i, try minimums near d_i, 2*d_i, 3*d_i, ...
    # But limit the search window to best_span - 1 (we need to beat current best)
    
    for i, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Try multiples of d as candidate minimums
        # Start from d, go up, but stop when m > max_val - best_span
        m = d
        while m <= max_val:
            if time.time() > deadline:
                break
            # Try to assign all divisors with minimum m
            # m must be assigned to divisor i
            used = {m}
            result = [0] * n
            result[i] = m
            success = True
            max_val_assigned = m
            
            for j in range(n):
                if j == i:
                    continue
                dj = divisors[j]
                mults_j = multiples[j]
                # Find smallest multiple of dj >= m, not in used, and < m + best_span
                lo, hi = 0, len(mults_j)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if mults_j[mid] < m:
                        lo = mid + 1
                    else:
                        hi = mid
                found = False
                for idx in range(lo, len(mults_j)):
                    v = mults_j[idx]
                    if v >= m + best_span:
                        break
                    if v not in used:
                        used.add(v)
                        result[j] = v
                        if v > max_val_assigned:
                            max_val_assigned = v
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                span = max_val_assigned - m
                if span < best_span:
                    best_span = span
                    best_result = result[:]
            
            m += d
    
    # Final polish: local search around the best result
    if best_result is not None and time.time() < deadline:
        # Try shifting each element slightly to reduce span
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                d = divisors[i]
                cur = best_result[i]
                # Try nearby multiples
                for offset in [-1, 1, -2, 2]:
                    new_val = cur + offset * d
                    if new_val < 1 or new_val > max_val:
                        continue
                    # Check if new_val is distinct from others
                    if new_val in best_result[:i] + best_result[i+1:]:
                        continue
                    trial = best_result[:]
                    trial[i] = new_val
                    trial_span = max(trial) - min(trial)
                    if trial_span < best_span:
                        best_span = trial_span
                        best_result = trial
                        improved = True
                        break
            if not improved:
                break
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best_result]
    return result

# EVOLVE-BLOCK-END
