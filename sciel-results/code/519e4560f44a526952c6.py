import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute divisor counts for heuristic sorting
    div_counts = [0] * (N + 1)
    for i in range(2, N + 1):
        for j in range(i, N + 1, i):
            div_counts[j] += 1

    def is_valid_addition_fast(a, current_set):
        """Check if adding 'a' to 'current_set' creates any bad triple."""
        if len(current_set) < 2:
            return True
            
        cs = current_set
        n = len(cs)
        
        for i in range(n):
            b = cs[i]
            for j in range(i + 1, n):
                c = cs[j]
                # Check a | b*c
                if (b * c) % a == 0:
                    return False
                # Check b | a*c
                if (a * c) % b == 0:
                    return False
                # Check c | a*b
                if (a * b) % c == 0:
                    return False
        return True

    def greedy_search(order):
        s = []
        for a in order:
            if is_valid_addition_fast(a, s):
                s.append(a)
        return s

    def local_search_improve(s):
        """Try to improve set s by swapping elements or adding/removing."""
        s = list(s)
        current_size = len(s)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try adding elements that weren't in the set
            for a in candidates:
                if a in s:
                    continue
                if is_valid_addition_fast(a, s):
                    s.append(a)
                    improved = True
                    break
            
            # Try swapping: remove one element, add another
            if not improved:
                for i, b in enumerate(s):
                    s_temp = s[:i] + s[i+1:]
                    for a in candidates:
                        if a in s_temp:
                            continue
                        if is_valid_addition_fast(a, s_temp):
                            s = s_temp + [a]
                            improved = True
                            break
                    if improved:
                        break
        
        return s

    best_set = []
    
    # Strategy 1: Upper Half (High probability of being valid and large)
    upper_half = list(range(N // 2 + 1, N + 1))
    # Verify upper half
    valid_upper = True
    uh_set = set(upper_half)
    for i in range(len(upper_half)):
        a = upper_half[i]
        for j in range(i + 1, len(upper_half)):
            b = upper_half[j]
            for k in range(j + 1, len(upper_half)):
                c = upper_half[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    valid_upper = False
                    break
            if not valid_upper:
                break
        if not valid_upper:
            break
            
    if valid_upper:
        best_set = upper_half[:]
    else:
        # Fallback to greedy if upper half is invalid (it shouldn't be, but just in case)
        strategies = [
            sorted(candidates, reverse=True),
            sorted(candidates, key=lambda x: div_counts[x]),
            sorted(candidates, key=lambda x: (div_counts[x], -x))
        ]
        for order in strategies:
            s = greedy_search(order)
            if len(s) > len(best_set):
                best_set = s

    # Local search on the best set found
    if time.time() < deadline - 10.0:
        improved = local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # Randomized restarts with local search if time permits
    while time.time() < deadline - 5.0:
        order = candidates[:]
        random.shuffle(order)
        s = greedy_search(order)
        
        if time.time() < deadline - 5.0:
            s = local_search_improve(s)
        
        if len(s) > len(best_set):
            best_set = s

    return [int(x) for x in best_set]
