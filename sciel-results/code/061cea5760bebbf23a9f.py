import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    
    # We need to find 10 distinct positive integers <= 1000000
    # such that representative[i] is divisible by divisors[i]
    # and we minimize max(reps) - min(reps)
    
    # Strategy: search over window size W from small to large
    # For each W, try to find a window [L, L+W] where each divisor
    # has at least one multiple in the window, and we can assign distinct multiples
    
    # A smarter approach: 
    # 1. For a given window [L, L+W], for each divisor d_i, the set of valid
    #    multiples in the window is {k*d_i : L <= k*d_i <= L+W}
    # 2. We need to pick one from each set such that all are distinct
    # 3. This is a bipartite matching / assignment problem
    
    # Even smarter: since divisors are fixed, let's think about what windows work.
    # The key insight: if W >= max(divisors) - 1, then every window of size W+1
    # contains at least one multiple of each divisor? No, that's not quite right.
    # Actually, in any window of length d, there's at least one multiple of d.
    # So in a window of length W, there's at least one multiple of d if W >= d-1.
    
    # But we need distinctness. Let's use a different approach:
    # Binary search on W, and for each W, scan possible L values.
    
    # For a given W, what L values could work?
    # L can be from 1 to MAX_V - W
    # But we can be smarter: L only needs to be checked at "interesting" positions.
    # The multiples change when L crosses a multiple of some divisor.
    
    # Let's think about it differently. For each divisor d_i, the multiples in [L, L+W]
    # are determined by floor((L+W)/d_i) and ceil(L/d_i).
    # The number of multiples is max(0, floor((L+W)/d_i) - ceil(L/d_i) + 1).
    
    # Key observation: as L increases by 1, the set of multiples for each divisor
    # changes only when L crosses a multiple of that divisor. So the "state" 
    # (which multiples are available) changes at O(n * MAX_V / d_i) points.
    
    # Alternative: for small W, the number of candidates per divisor is small,
    # so we can do backtracking.
    
    # Let me try a direct approach:
    # For W from 1 upward:
    #   For L in a smart set of starting points:
    #     Check if each divisor has at least one multiple in [L, L+W]
    #     If yes, try to assign distinct multiples via backtracking
    
    # What's a good upper bound on W? The divisors are at most 39.
    # In the worst case, W might need to be larger due to distinctness constraints.
    # But likely W is small, maybe < 100 or so.
    
    best_val = float('inf')
    best_assign = None
    
    def try_window(L, W):
        """Try to assign distinct multiples in [L, L+W] for each divisor."""
        if time.time() > deadline:
            return None
        
        # For each divisor, find all multiples in [L, L+W]
        candidates = []
        for d in divisors:
            # Smallest multiple >= L
            k_min = (L + d - 1) // d  # ceil(L/d)
            k_max = (L + W) // d       # floor((L+W)/d)
            if k_min > k_max:
                return None  # No multiple of d in window
            mults = [k * d for k in range(k_min, k_max + 1)]
            candidates.append(mults)
        
        # Now we need to pick one from each list such that all are distinct
        # Use backtracking
        used = set()
        assignment = [None] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            for val in candidates[idx]:
                if val not in used:
                    used.add(val)
                    assignment[idx] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assignment[idx] = None
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Search over W from 1 upward
    # For each W, we need to find if there exists L such that try_window(L, W) succeeds
    # L ranges from 1 to MAX_V - W
    
    # To speed up, note that the "interesting" L values are near multiples of the divisors.
    # Let's collect all "critical" L values: L such that L or L+W is near a multiple of some d.
    # Actually, simpler: just scan L in steps, but be smart.
    
    # The state of "which multiples are available" for each divisor changes when
    # L crosses a multiple of d, or when L+W crosses a multiple of d.
    # So critical points are: k*d for all d, k*d - W for all d.
    
    # For a given W, let's generate candidate L values around these critical points.
    
    for W in range(1, 200):
        if time.time() > deadline:
            break
        
        # Generate candidate L values
        # L must be in [1, MAX_V - W]
        # Critical points: for each divisor d, multiples k*d and k*d - W
        # We want L such that the set of available multiples is "stable" or at boundaries
        
        # Simple approach: try L = 1, and then L near each critical point
        # But that could be many points. Let's try a coarser approach first.
        
        # Try L values that are multiples of the LCM of all divisors? No, that's huge.
        
        # Let's try: for each divisor d, the pattern of available multiples repeats
        # with period d. The combined pattern has period lcm(all divisors).
        # But that's too large.
        
        # Alternative: just try a range of L values. For small W, the window is small,
        # so we can afford to check more L values.
        
        # Let's try L from 1 to min(MAX_V - W, 10000) first, in steps of 1.
        # If W is small, this should be fast enough.
        
        max_L_to_try = min(MAX_V - W, 50000)
        
        # But 50000 * 200 = 10 million windows, each with backtracking... too slow.
        
        # Better: for each W, only try L values where the "pattern" changes.
        # The pattern for divisor d changes at L = k*d and L = k*d - W.
        # So for a given W, collect all such critical L values, sort them,
        # and test one L from each "interval" between consecutive critical points.
        
        criticals = set()
        criticals.add(1)
        criticals.add(max_L_to_try)
        
        for d in divisors:
            # Multiples of d: k*d, where k*d is in [1, MAX_V]
            # We care about L such that L is near k*d or k*d - W
            k_start = max(1, (1 + W) // d)  # k*d - W >= 1 => k*d >= 1+W => k >= ceil((1+W)/d)
            k_end = MAX_V // d
            for k in range(k_start, k_end + 1):
                m = k * d
                # L = m: window [m, m+W] includes m
                # L = m - W: window [m-W, m] includes m
                if 1 <= m <= max_L_to_try:
                    criticals.add(m)
                if 1 <= m - W <= max_L_to_try:
                    criticals.add(m - W)
        
        # Also add L = m - 1 and L = m + 1 for safety
        crit_list = sorted(criticals)
        # Test one L from each interval
        test_Ls = set()
        for i in range(len(crit_list)):
            test_Ls.add(crit_list[i])
        # Also add midpoints
        for i in range(len(crit_list) - 1):
            mid = (crit_list[i] + crit_list[i+1]) // 2
            if 1 <= mid <= max_L_to_try:
                test_Ls.add(mid)
        
        test_Ls = sorted(test_Ls)
        
        found = False
        for L in test_Ls:
            if time.time() > deadline:
                break
            if L < 1 or L > MAX_V - W:
                continue
            assign = try_window(L, W)
            if assign is not None:
                val = max(assign) - min(assign)
                if val < best_val:
                    best_val = val
                    best_assign = assign[:]
                found = True
                # We found a solution for this W, so we can break (W is increasing)
                break
        
        if found:
            # We found the minimum W that works. But maybe a larger W gives a smaller
            # max-min? No, if W is the window size, max-min <= W. So smaller W is better.
            # Actually, we're minimizing max-min, and W is an upper bound on max-min.
            # So the first W that works gives us a solution with max-min <= W.
            # But maybe within that W, we can find a tighter assignment.
            # Since we're iterating W from 1 up, the first W that works is optimal
            # in terms of the window size, but the actual max-min might be less than W.
            # Let's continue searching for a bit to see if we can do better.
            # Actually, no: if W=5 works, then max-min <= 5. If W=4 doesn't work,
            # then no solution with max-min <= 4 exists. So W is the answer.
            # But wait: we're checking if there EXISTS a window of size W that works.
            # The actual max-min of the assignment might be less than W.
            # But we want to minimize max-min, so we should find the assignment with
            # the smallest max-min, not just any assignment in a window of size W.
            
            # Hmm, let me reconsider. We're searching over W (window size). For each W,
            # we find if there's a window [L, L+W] with a valid assignment. The max-min
            # of that assignment is at most W. So the minimal W that works gives us
            # a solution with max-min <= W. But the actual max-min might be smaller.
            
            # To find the true minimum, we should search more carefully. But for now,
            # let's take this as a good starting point and potentially refine.
            
            # Actually, I realize my approach is slightly off. Let me reconsider.
            # We want to minimize max(reps) - min(reps). Let's call this range R.
            # If we find an assignment with range R, then it fits in a window of size R.
            # So the minimum R is the minimum W for which a valid window exists.
            # So iterating W from 1 up and taking the first W that works gives us the
            # minimum possible range. And the assignment we find has range <= W.
            # But we should verify that the actual range of our assignment is indeed W
            # or less, and that we can't do better.
            
            # Since we're iterating W in increasing order, the first W that works is
            # the minimum possible range. So we can stop here.
            break
    
    if best_assign is None:
        # Fallback: just use the divisors themselves (which are distinct and <= 1000000)
        # But they might not be optimal. Let's at least return something valid.
        best_assign = divisors[:]
    
    # Ensure all are Python ints
    result = [int(x) for x in best_assign]
    return result
