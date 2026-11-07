import time
from math import gcd
from functools import reduce
from bisect import bisect_left, bisect_right

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Build merged sorted list of all (value, divisor_index) pairs
    # Each multiple belongs to one or more divisors, but we treat each (value, i) as a distinct entry
    # Actually, a value can be a multiple of multiple divisors. We need to track which divisors each value covers.
    # For the sliding window, we want: for each value in the window, which divisors does it satisfy?
    # Then we need to check if the union of satisfied divisors covers all 10.
    # But we also need distinctness: each divisor gets a DIFFERENT value.
    # So it's not just "cover all divisors" - we need a matching.
    
    # Alternative approach: 
    # 1. Generate all multiples for each divisor.
    # 2. Create a merged sorted list of all unique values, with a bitmask of which divisors each value satisfies.
    # 3. Use sliding window to find the shortest window where we can assign distinct values to all 10 divisors.
    
    # For step 3, the assignment check is expensive. Let's think differently.
    
    # Key insight: If we have a window [L, R], and for each divisor i, the set of multiples in [L,R] is non-empty,
    # AND there exists a system of distinct representatives (SDR), then we have a valid solution.
    # The Hall's condition must hold: for every subset S of divisors, the union of their candidate sets has size >= |S|.
    # But checking Hall's condition is expensive.
    
    # Simpler: Since we only need 10 divisors, we can use backtracking for the SDR check,
    # but we want to do it efficiently during the sliding window.
    
    # Let me try a different strategy:
    # 1. For each possible window width W (from small to large), check if any window of width W works.
    # 2. To check a window efficiently, use the merged list approach.
    
    # Actually, let me try the direct sliding window approach:
    # - Create a sorted list of all (value, divisor_mask) where divisor_mask is a bitmask of divisors that value is a multiple of.
    # - Use two pointers (left, right) to maintain a window.
    # - For the window, check if there's a valid SDR.
    # - To check SDR: for each divisor, collect its candidates in the window, then try to find a matching.
    
    # The challenge is efficiency. Let's estimate:
    # Total multiples: sum(10^6/d for d in divisors) ≈ 10^6 * (1/6+1/10+1/14+1/15+1/21+1/22+1/26+1/33+1/35+1/39)
    # ≈ 10^6 * (0.167+0.1+0.071+0.067+0.048+0.045+0.038+0.030+0.029+0.026)
    # ≈ 10^6 * 0.621 ≈ 621,000 entries.
    
    # For each window position, checking SDR via backtracking on 10 divisors with small candidate sets
    # should be fast if the candidate sets are small.
    
    # But the number of window positions is ~621,000, and for each we do a backtracking search.
    # This might be too slow.
    
    # Better: Use the fact that we're looking for the MINIMUM window.
    # We can binary search on W, and for a given W, check if any window of width W works.
    # For a given W, we can slide the window and check each position.
    # But we need to check all positions for a given W, which is still ~621,000 checks.
    
    # Alternative: For a given W, we can jump to the next relevant position (where the set of candidates changes).
    # This is what the current program does, but it's complex.
    
    # Let me try a simpler approach first: 
    # 1. Pre-compute for each divisor, its multiples.
    # 2. Try small W values (1, 2, 3, ...) and for each W, check all possible L positions.
    # 3. For each (W, L), check if a valid SDR exists.
    # 4. Return the first valid (W, L) found.
    
    # To make this efficient:
    # - For a given W, the number of L positions to check is bounded by the number of "events" where the candidate set changes.
    # - We can jump L to the next event.
    
    # Let me implement the event-driven approach more carefully.
    
    # Pre-compute multiples for each divisor
    multiples = []
    for d in divisors:
        m = []
        x = d
        while x <= max_val:
            m.append(x)
            x += d
        multiples.append(m)
    
    # Helper: Check if a valid distinct assignment exists for candidates
    def check_assignment(candidates):
        for c in candidates:
            if not c:
                return None
        
        # Sort divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        if len(candidates[order[0]]) == 0:
            return None
            
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
        
        if backtrack(0):
            return assignment
        return None

    # For a given W, find if any window of width W has a valid SDR
    def find_solution_for_W(W):
        if W >= max_val:
            return None
        
        # We'll iterate L, jumping to the next event where the candidate set changes.
        # Event: when a multiple enters or leaves the window [L, L+W].
        
        L = 1
        while L <= max_val - W:
            # Build candidates for current L
            candidates = [[] for _ in range(n)]
            for i in range(n):
                left_idx = bisect_left(multiples[i], L)
                right_idx = bisect_right(multiples[i], L + W)
                candidates[i] = multiples[i][left_idx:right_idx]
                
            # Check assignment
            sol = check_assignment(candidates)
            if sol is not None:
                return sol
            
            # Determine next L where candidate set changes
            next_L = max_val + 1
            
            for i in range(n):
                # A multiple m leaves the window when L > m, i.e., L >= m+1
                # A multiple m enters the window when L+W >= m, i.e., L >= m-W
                # The next change is the minimum of:
                # - (m+1) for m in current window (leaving)
                # - (m-W) for m just outside window (entering)
                
                if candidates[i]:
                    # First multiple in window will leave next
                    first_in = candidates[i][0]
                    next_L = min(next_L, first_in + 1)
                
                # First multiple outside window (greater than L+W) will enter
                idx = bisect_right(multiples[i], L + W)
                if idx < len(multiples[i]):
                    next_m = multiples[i][idx]
                    next_L = min(next_L, next_m - W)
            
            if next_L <= L:
                next_L = L + 1
            
            L = next_L
            
        return None

    # Try W from 1 upwards
    best_solution = None
    for W in range(1, 200):
        if time.time() > deadline:
            break
            
        sol = find_solution_for_W(W)
        if sol is not None:
            best_solution = sol
            break
        
    if best_solution is not None:
        return [int(x) for x in best_solution]
    
    # Fallback
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
