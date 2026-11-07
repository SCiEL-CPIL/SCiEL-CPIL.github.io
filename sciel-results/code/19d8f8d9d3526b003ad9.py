import math
import time
from collections import defaultdict
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # L is 30030
    
    # Generate all multiples in [0, L)
    events = []
    for i, d in enumerate(divisors):
        for m in range(0, L, d):
            events.append((m, i))
            
    events.sort(key=lambda x: x[0])
    
    # Extended events for circular handling
    extended_events = events + [(v + L, i) for v, i in events]
    N_ext = len(extended_events)
    
    # First, find the minimum covering span (theoretical lower bound)
    counts = [0] * n
    unique_count = 0
    left = 0
    best_covering_span = float('inf')
    
    for right in range(N_ext):
        val_r, idx_r = extended_events[right]
        counts[idx_r] += 1
        if counts[idx_r] == 1:
            unique_count += 1
            
        while unique_count == n:
            val_l, idx_l = extended_events[left]
            span = val_r - val_l
            if span < best_covering_span:
                best_covering_span = span
                
            counts[idx_l] -= 1
            if counts[idx_l] == 0:
                unique_count -= 1
            left += 1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds to be safe
    
    # Bipartite matching: left nodes are divisors, right nodes are candidate values
    # For a given window, check if we can assign distinct values to each divisor
    
    def can_assign_matching(vals_by_idx):
        """
        Check if there's a system of distinct representatives.
        vals_by_idx[i] is a list of candidate values for divisor i.
        Returns the assignment as a dict {divisor_idx: value} or None.
        """
        # Build adjacency: for each value, which divisors can use it
        # Actually, for matching: left = divisors, right = values
        # But values can be shared, so we need to track which values are used
        
        # Simpler: use augmenting path
        # match_val[v] = which divisor is matched to value v
        match_div = [-1] * n  # match_div[i] = value assigned to divisor i, or -1
        match_val = {}  # match_val[v] = divisor index matched to value v
        
        def try_augment(div_idx, visited):
            for v in vals_by_idx[div_idx]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_val or try_augment(match_val[v], visited):
                    match_div[div_idx] = v
                    match_val[v] = div_idx
                    return True
            return False
        
        # Try to find augmenting path for each divisor
        for i in range(n):
            visited = set()
            if not try_augment(i, visited):
                return None
        
        # All divisors matched
        return {i: match_div[i] for i in range(n)}
    
    def find_distinct_assignment(S):
        """Find a distinct assignment with span <= S, or return None."""
        counts = [0] * n
        unique_count = 0
        left = 0
        
        for right in range(N_ext):
            val_r, idx_r = extended_events[right]
            counts[idx_r] += 1
            if counts[idx_r] == 1:
                unique_count += 1
                
            # Shrink left until window length > S
            while val_r - extended_events[left][0] > S:
                val_l, idx_l = extended_events[left]
                counts[idx_l] -= 1
                if counts[idx_l] == 0:
                    unique_count -= 1
                left += 1
                
            if unique_count == n:
                # Current window [left, right] has span <= S and covers all.
                # Extract values for each divisor in this window.
                vals_by_idx = defaultdict(list)
                for k in range(left, right + 1):
                    v, i = extended_events[k]
                    vals_by_idx[i].append(v)
                
                # Check if distinct assignment exists using bipartite matching
                assignment = can_assign_matching(vals_by_idx)
                if assignment is not None:
                    return [assignment[i] for i in range(n)]
            
            # Deadline check
            if time.time() > deadline:
                return None
                
        return None
    
    # Try increasing S from the covering span
    best_result = None
    for S in range(int(best_covering_span), int(best_covering_span) + 200):
        if time.time() > deadline:
            break
        res = find_distinct_assignment(S)
        if res is not None:
            # Verify distinctness and divisibility
            if len(set(res)) == n:
                valid = True
                for i, v in enumerate(res):
                    if not isinstance(v, int) or isinstance(v, bool):
                        valid = False
                        break
                    if v <= 0 or v > 1000000:
                        valid = False
                        break
                    if v % divisors[i] != 0:
                        valid = False
                        break
                if valid:
                    best_result = [int(v) for v in res]
                    break
    
    if best_result is not None:
        return best_result
    
    # Fallback: should not reach here, but return a valid set if possible
    # Try a larger range
    for S in range(int(best_covering_span) + 200, int(best_covering_span) + 500):
        if time.time() > deadline:
            break
        res = find_distinct_assignment(S)
        if res is not None:
            if len(set(res)) == n:
                valid = True
                for i, v in enumerate(res):
                    if not isinstance(v, int) or isinstance(v, bool):
                        valid = False
                        break
                    if v <= 0 or v > 1000000:
                        valid = False
                        break
                    if v % divisors[i] != 0:
                        valid = False
                        break
                if valid:
                    return [int(v) for v in res]
    
    # Last resort: construct a valid (but not optimal) solution
    # Use multiples that are guaranteed to be distinct
    result = []
    for i, d in enumerate(divisors):
        # Find a multiple of d that is not in result
        m = d
        while m in result or m > 1000000:
            m += d
        result.append(int(m))
    
    return result

if __name__ == '__main__':
    res = construct()
    print(res)
