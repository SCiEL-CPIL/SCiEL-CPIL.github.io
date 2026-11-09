import time
import random
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute conflict information
    # For a valid set S: for every a < b <= c in S, a must NOT divide (b+c)
    # We can think of this as: for each triple (a,b,c) with a<b<=c, if a|(b+c) then we can't have all three in S.
    # This is a 3-uniform hypergraph independent set problem.
    
    # Strategy: Greedy construction from upper tail, then local search with smarter moves.
    # Key insight: numbers in the upper half [n/2+1, n] can never be 'a' in a violation 
    # because b+c >= 2*(n/2+1) > n, and a <= n/2, so a can't divide b+c if b+c > 2a... 
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, if a > (b+c)/2, then a cannot divide b+c (since the only multiple would be b+c itself, 
    # but a < b <= c means a < b+c, and a > (b+c)/2 means a doesn't divide b+c unless b+c = a, impossible).
    # So if a > (b+c)/2, no conflict.
    
    # For a in the upper half [129, 256]: if b >= a+1 >= 130, then b+c >= 260, and a <= 256.
    # a divides b+c means b+c = k*a for some k >= 1. Since b+c >= 260 and a >= 129, k could be 1 or 2.
    # k=1: b+c = a, impossible since b > a.
    # k=2: b+c = 2a, possible if b+c = 2a.
    # So for a in [129,256], conflicts only arise when b+c = 2a exactly (k=2), or possibly k=1 (impossible).
    
    # Let me just build the conflict structure properly.
    
    # For each pair (b,c) with b <= c, find all a < b such that a | (b+c).
    # Then the constraint is: for each such (a,b,c), not all three can be in S.
    
    # Greedy approach: Start with a candidate set and iteratively improve.
    
    def check_valid(S):
        """Check if set S is valid."""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if j <= i:
                    continue
                for k, c in enumerate(S_list):
                    if k < j:
                        continue
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if adding x to valid set S keeps it valid."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Case 1: x plays role of 'a' (smallest). Need b,c in S with x < b <= c and x | (b+c)
        for idx_b, b in enumerate(S_list):
            if b <= x:
                continue
            for c in S_list[idx_b:]:  # c >= b
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x plays role of 'b' (middle). Need a,c in S with a < x <= c and a | (x+c)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x plays role of 'c' (largest). Need a,b in S with a < b <= x and a | (b+x)
        for a in S_list:
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def greedy_construct(threshold, order):
        """Greedy construction starting from [threshold, 256], adding from order."""
        S = set(range(threshold, 257))
        for candidate in order:
            if time.time() > deadline - 5:
                break
            if candidate in S:
                continue
            if can_add(S, candidate):
                S.add(candidate)
        return S
    
    def local_search(S, max_rounds=20):
        """Local search: try to add elements, or swap to enable additions."""
        S = set(S)
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline - 10:
                break
            
            # Try to add any element not in S
            added = False
            for x in range(1, 257):
                if time.time() > deadline - 10:
                    break
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    added = True
                    improved = True
            
            if not added:
                # Try single swaps: remove one, add another (same size, but may enable further additions)
                S_list = sorted(S)
                for a in S_list:
                    if time.time() > deadline - 10:
                        break
                    S_temp = S - {a}
                    for x in range(1, 257):
                        if time.time() > deadline - 10:
                            break
                        if x in S:
                            continue
                        if can_add(S_temp, x):
                            # Check if this swap enables adding more elements
                            S_new = S_temp | {x}
                            # Try to add more
                            count_before = len(S)
                            for y in range(1, 257):
                                if y in S_new:
                                    continue
                                if can_add(S_new, y):
                                    S_new.add(y)
                            if len(S_new) > count_before:
                                S = S_new
                                improved = True
                                break
                    if improved:
                        break
        
        return S
    
    # Initial best: just the upper half
    best = set(range(129, 257))
    best_size = len(best)
    
    # Try multiple greedy constructions with different thresholds and orders
    thresholds = [129, 140, 150, 160, 170, 180, 190, 200]
    
    orders = [
        list(range(1, 129)),                          # ascending
        list(range(128, 0, -1)),                       # descending
        list(range(2, 129, 2)),                        # evens first
        list(range(1, 129, 2)),                        # odds first
        sorted(range(1, 129), key=lambda x: -x.bit_count()),  # more bits first
        sorted(range(1, 129), key=lambda x: x.bit_count()),   # fewer bits first
        list(range(1, 129, 3)),                        # multiples of 3
    ]
    
    for threshold in thresholds:
        if time.time() > deadline - 20:
            break
        for order in orders:
            if time.time() > deadline - 15:
                break
            adjusted_order = [x for x in order if x < threshold]
            if not adjusted_order:
                continue
            S = greedy_construct(threshold, adjusted_order)
            if len(S) > best_size:
                best = S
                best_size = len(S)
    
    # Local search on the best
    if time.time() < deadline - 25:
        best = local_search(best, max_rounds=30)
        best_size = len(best)
    
    # Try local search on other promising solutions
    if time.time() < deadline - 20:
        for threshold in [129, 140, 150, 160]:
            if time.time() > deadline - 15:
                break
            S = greedy_construct(threshold, list(range(1, threshold)))
            if len(S) >= best_size - 5:
                S = local_search(S, max_rounds=20)
                if len(S) > best_size:
                    best = S
                    best_size = len(S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 5:
        S_list = sorted(best)
        for x in range(1, 257):
            if time.time() > deadline - 5:
                break
            if x in best:
                continue
            if can_add(best, x):
                best.add(x)
    
    # Verify validity
    if not check_valid(best):
        # Fallback: use a known valid set
        best = set(range(129, 257))
    
    return sorted(best)
