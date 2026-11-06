import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55
    
    def has_3ap_with_x(s, x):
        # s is a set of ints, x is an int not in s
        # Returns True if adding x to s creates a 3-AP
        for a in s:
            # x is middle: a, x, b => b = 2x - a
            b = 2 * x - a
            if b != a and b in s:
                return True
            # x is end: a, b, x or x, a, b
            # If x is max: a, b, x => b = (a+x)/2 => 2b = a+x => a = 2b-x.
            # If x is min: x, a, b => a = x+d, b = x+2d => b = 2a-x.
            # So in both cases, we check if 2a-x is in s.
            # Note: if 2a-x == x, then a=x, impossible.
            # if 2a-x == a, then a=x, impossible.
            # So we just check if 2a-x is in s.
            b_end = 2 * a - x
            if b_end in s:
                return True
        return False

    def greedy_construct():
        s = set()
        for x in range(1, 257):
            if not has_3ap_with_x(s, x):
                s.add(x)
        return s

    def local_search(s, start_time):
        best_set = set(s)
        best_sum = sum(Fraction(1, x) for x in best_set)
        all_candidates = set(range(1, 257))
        
        while time.time() < start_time + 20: # 20s per restart?
            improved = False
            
            # 1. Add
            # Sort candidates by value (smallest x first)
            to_add = sorted(all_candidates - best_set)
            for x in to_add:
                if time.time() >= start_time + 20: break
                if not has_3ap_with_x(best_set, x):
                    best_set.add(x)
                    best_sum += Fraction(1, x)
                    improved = True
            if improved:
                continue
                
            # 2. Swap
            # Remove largest (lowest value)
            to_remove = sorted(best_set, reverse=True)
            for rem in to_remove:
                if time.time() >= start_time + 20: break
                
                # Find best x to add such that x < rem and valid
                # x < rem ensures 1/x > 1/rem
                best_add = None
                best_gain = Fraction(0)
                
                # Iterate x from 1 to rem-1
                for x in range(1, rem):
                    if time.time() >= start_time + 20: break
                    if x in best_set:
                        continue
                    if not has_3ap_with_x(best_set - {rem}, x):
                        gain = Fraction(1, x) - Fraction(1, rem)
                        if gain > best_gain:
                            best_gain = gain
                            best_add = x
                
                if best_add is not None:
                    best_set.remove(rem)
                    best_set.add(best_add)
                    best_sum += best_gain
                    improved = True
                    break # Restart loop
            
            if not improved:
                break
                
        return best_set, best_sum

    best_overall_set = set()
    best_overall_sum = Fraction(0)
    
    # Run multiple restarts
    # Since greedy is deterministic, we need to vary it or use random kicks.
    # For now, just run greedy + local search.
    # If we have time, we can try random initial sets or random swaps.
    
    # 1st run: Standard Greedy
    s0 = greedy_construct()
    s1, sum1 = local_search(s0, deadline)
    if sum1 > best_overall_sum:
        best_overall_set = s1
        best_overall_sum = sum1
        
    # If time remains, try a different strategy?
    # e.g. Start with powers of 2?
    if time.time() < deadline - 5:
        s_powers = set()
        x = 1
        while x <= 256:
            s_powers.add(x)
            x *= 2
        s2, sum2 = local_search(s_powers, deadline)
        if sum2 > best_overall_sum:
            best_overall_set = s2
            best_overall_sum = sum2
            
    # If time remains, try random restarts
    while time.time() < deadline - 5:
        # Random kick: start with greedy, then remove 5 random elements
        s_rand = greedy_construct()
        if len(s_rand) > 5:
            items = list(s_rand)
            for _ in range(5):
                if items:
                    idx = int(time.time() * 1000) % len(items)
                    s_rand.remove(items[idx])
                    items.pop(idx)
        
        s3, sum3 = local_search(s_rand, deadline)
        if sum3 > best_overall_sum:
            best_overall_set = s3
            best_overall_sum = sum3
            
    return sorted(best_overall_set)
