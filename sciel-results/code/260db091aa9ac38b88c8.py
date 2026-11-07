import time
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
DEADLINE = 85.0  # safety margin from 90s

def _lcm(a, b):
    return a * b // gcd(a, b)

def _all_lcm():
    result = 1
    for d in DIVISORS:
        result = _lcm(result, d)
    return result

LCM_ALL = _all_lcm()  # 5005000

def _matching_exists(L, W):
    """Check if there's a perfect matching: assign each divisor a distinct multiple in [L, L+W]."""
    adj = []
    for i in range(N):
        d = DIVISORS[i]
        first = ((L + d - 1) // d) * d
        candidates = []
        v = first
        while v <= L + W:
            candidates.append(v)
            v += d
        adj.append(candidates)
    
    for i in range(N):
        if not adj[i]:
            return False, None
    
    match_right = {}
    
    def try_kuhn(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_right or try_kuhn(match_right[v], visited):
                match_right[v] = i
                return True
        return False
    
    for i in range(N):
        visited = set()
        if not try_kuhn(i, visited):
            return False, None
    
    assignment = [None] * N
    for v, i in match_right.items():
        assignment[i] = v
    
    return True, assignment

def construct():
    start_time = time.time()
    
    # Step 1: Collect all multiples of each divisor in [1, MAX_VAL]
    # For each value, record which divisors it's a multiple of
    # We'll build a sorted list of (value, divisor_indices) pairs
    
    # For efficiency, for each divisor d, generate multiples and tag them
    # Then sort by value and merge
    
    # Actually, let's use a different approach:
    # Create a list of all candidate values with their divisor memberships
    # But that could be huge. Instead:
    
    # Better: For the sliding window, we need to know for each position in the
    # sorted list of all multiples, which divisors are covered.
    
    # Approach: Generate all multiples, sort by value, then slide.
    # For each divisor d, multiples are d, 2d, 3d, ...
    # Total count: sum(MAX_VAL/d for d in DIVISORS) ≈ 166667+100000+71429+66667+47619+45455+38462+30303+28571+25641 ≈ 520814
    
    # Build sorted list of (value, divisor_index)
    # But a value can be a multiple of multiple divisors
    
    # For sliding window: we need to track, for the current window [left, right],
    # how many multiples of each divisor are in the window.
    # We use a frequency array of size N.
    
    # Build events: for each divisor i, for each multiple m of DIVISORS[i] in [1, MAX_VAL],
    # add an event at position m with divisor index i.
    # Sort all events by value.
    # Then slide: maintain count[i] = number of multiples of divisor i in current window.
    # When all count[i] >= 1, we have a valid window. Shrink from left.
    
    # But we need DISTINCT values. The sliding window gives us a set of values where
    # each divisor has at least one representative. We then need to verify we can assign
    # distinct values.
    
    # For now, let's find the minimal span windows and verify with matching.
    
    # Generate all (value, divisor_index) pairs
    events = []
    for i, d in enumerate(DIVISORS):
        v = d
        while v <= MAX_VAL:
            events.append((v, i))
            v += d
    
    events.sort()  # sort by value, then by divisor index
    
    # Sliding window
    n_events = len(events)
    counts = [0] * N
    left = 0
    best_span = float('inf')
    best_windows = []  # list of (span, left_idx, right_idx)
    
    for right in range(n_events):
        counts[events[right][1]] += 1
        
        # Try to shrink from left while all counts >= 1
        while all(c >= 1 for c in counts):
            cur_span = events[right][0] - events[left][0]
            if cur_span < best_span:
                best_span = cur_span
                best_windows = [(cur_span, left, right)]
            elif cur_span == best_span:
                best_windows.append((cur_span, left, right))
            
            counts[events[left][1]] -= 1
            left += 1
    
    # Now verify the best windows with matching
    # But best_windows might have many entries with same span
    # Let's try the top few
    
    if time.time() - start_time > DEADLINE:
        # Fallback
        return DIVISORS[:]
    
    # Try to find a valid assignment for the best span
    # For each candidate window, check if matching exists
    # The window is [events[left][0], events[right][0]]
    
    # Deduplicate windows by (L, W)
    tried = set()
    for span, li, ri in best_windows[:100]:
        if time.time() - start_time > DEADLINE:
            break
        L = events[li][0]
        W = span
        key = (L, W)
        if key in tried:
            continue
        tried.add(key)
        
        ok, assignment = _matching_exists(L, W)
        if ok:
            vals = list(assignment)
            if len(set(vals)) == N and all(1 <= v <= MAX_VAL for v in vals):
                result = [int(v) for v in vals]
                return result
    
    # If no exact match found, try nearby spans
    # Search W from best_span to best_span + 200
    for W in range(best_span, best_span + 201):
        if time.time() - start_time > DEADLINE:
            break
        # Try a few L values
        for L in range(1, min(5000, LCM_ALL) + 1):
            if L + W > MAX_VAL:
                break
            ok, assignment = _matching_exists(L, W)
            if ok:
                vals = list(assignment)
                if len(set(vals)) == N and all(1 <= v <= MAX_VAL for v in vals):
                    result = [int(v) for v in vals]
                    return result
    
    # Fallback
    return DIVISORS[:]
# EVOLVE-BLOCK-END
