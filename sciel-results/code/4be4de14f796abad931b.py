"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def is_compatible(x, S):
        """Check if x can be added to set S (list of ints)"""
        for y in S:
            if lcm(x, y) <= 256:
                return False
        return True
    
    def reciprocal_sum(S):
        return sum(1.0 / x for x in S)
    
    # Strategy: Multiple greedy initializations + local search with swaps
    
    best_set = []
    best_val = 0.0
    
    def local_search(S, max_iters=200):
        """Improve set by adding compatible elements and swapping for better sums"""
        S = list(S)
        S_set = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try adding each candidate not in S
            for x in range(2, 257):
                if x not in S_set:
                    if is_compatible(x, S):
                        S.append(x)
                        S_set.add(x)
                        improved = True
            
            # Try swapping: remove one element, add one or more that improve sum
            for i, y in enumerate(list(S)):
                if time.time() > deadline - 5:
                    return S
                # Try removing y and adding a single x
                S_without = [z for z in S if z != y]
                S_without_set = set(S_without)
                best_swap_gain = 0
                best_swap_x = None
                for x in range(2, 257):
                    if x not in S_without_set and x != y:
                        if is_compatible(x, S_without):
                            gain = 1.0/x - 1.0/y
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap_x = x
                if best_swap_x is not None and best_swap_gain > 0:
                    S.remove(y)
                    S_set.discard(y)
                    S.append(best_swap_x)
                    S_set.add(best_swap_x)
                    improved = True
            
            # Try removing two elements and adding one or two
            # This is expensive, so only do it if time permits
            if time.time() < deadline - 10:
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        y1, y2 = S_list[i], S_list[j]
                        S_without = [z for z in S if z != y1 and z != y2]
                        S_without_set = set(S_without)
                        base_gain = 1.0/y1 + 1.0/y2
                        # Try adding single x
                        for x in range(2, 257):
                            if x not in S_without_set and x not in (y1, y2):
                                if is_compatible(x, S_without):
                                    gain = 1.0/x - base_gain
                                    if gain > 0:
                                        S.remove(y1)
                                        S.remove(y2)
                                        S_set.discard(y1)
                                        S_set.discard(y2)
                                        S.append(x)
                                        S_set.add(x)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
        return S
    
    # Multiple greedy strategies for initialization
    strategies = []
    
    # Strategy A: Greedy by 1/x descending (smallest x first)
    S_a = []
    for x in range(2, 257):
        if is_compatible(x, S_a):
            S_a.append(x)
    strategies.append(S_a)
    
    # Strategy B: Start with large numbers, then add small
    S_b = []
    for x in range(256, 1, -1):
        if is_compatible(x, S_b):
            S_b.append(x)
    for x in range(2, 129):
        if is_compatible(x, S_b):
            S_b.append(x)
    strategies.append(S_b)
    
    # Strategy C: Numbers > 128 first, then small
    S_c = []
    for x in range(256, 128, -1):
        if is_compatible(x, S_c):
            S_c.append(x)
    for x in range(2, 129):
        if is_compatible(x, S_c):
            S_c.append(x)
    strategies.append(S_c)
    
    # Strategy D: Only odd numbers > 64 (likely compatible), then add small
    S_d = []
    for x in range(256, 64, -2):  # odd numbers
        if is_compatible(x, S_d):
            S_d.append(x)
    for x in range(2, 129):
        if is_compatible(x, S_d):
            S_d.append(x)
    strategies.append(S_d)
    
    # Evaluate all strategies with local search
    for S in strategies:
        S_improved = local_search(S, max_iters=50)
        val = reciprocal_sum(S_improved)
        if val > best_val:
            best_val = val
            best_set = S_improved
    
    # Random restarts with different orderings
    import random
    random.seed(42)
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        # Random ordering biased toward small numbers
        candidates = list(range(2, 257))
        random.shuffle(candidates)
        
        S_r = []
        for x in candidates:
            if time.time() > deadline - 15:
                break
            if is_compatible(x, S_r):
                S_r.append(x)
        
        S_r = local_search(S_r, max_iters=30)
        val = reciprocal_sum(S_r)
        if val > best_val:
            best_val = val
            best_set = S_r
    
    # Final: try to add any remaining compatible numbers
    final_S = list(best_set)
    final_set = set(final_S)
    for x in range(2, 257):
        if x not in final_set:
            if is_compatible(x, final_S):
                final_S.append(x)
                final_set.add(x)
    
    return final_S
# EVOLVE-BLOCK-END
