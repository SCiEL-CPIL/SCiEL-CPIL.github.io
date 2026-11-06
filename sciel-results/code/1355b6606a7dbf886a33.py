import time
from fractions import Fraction
import random

def construct():
    deadline = time.time() + 55
    random.seed(42)
    
    def has_3ap_with_x(s, x):
        # s is a set of ints, x is an int not in s
        # Returns True if adding x to s creates a 3-AP
        for a in s:
            # x is middle: a, x, b => b = 2x - a
            b = 2 * x - a
            if b != a and b in s:
                return True
            # x is end: check if 2a-x is in s
            b_end = 2 * a - x
            if b_end in s:
                return True
        return False

    def greedy_construct_priority():
        """Greedy that prioritizes smaller numbers (higher reciprocal value)"""
        s = set()
        # Try numbers in order of decreasing reciprocal value (increasing x)
        # But we want to maximize sum, so we should try to include small numbers first
        # However, small numbers create more AP constraints, so we need to be careful
        # Strategy: try numbers from 1 to 256, but if we get stuck, try a different order
        for x in range(1, 257):
            if not has_3ap_with_x(s, x):
                s.add(x)
        return s

    def greedy_construct_descending():
        """Greedy that tries larger numbers first (less restrictive)"""
        s = set()
        for x in range(256, 0, -1):
            if not has_3ap_with_x(s, x):
                s.add(x)
        return s

    def greedy_construct_mixed():
        """Mixed strategy: alternate between small and large"""
        s = set()
        candidates = list(range(1, 257))
        # Shuffle with bias toward smaller numbers
        random.shuffle(candidates)
        # Sort by a mixed criterion: prefer smaller but with some randomness
        for x in candidates:
            if not has_3ap_with_x(s, x):
                s.add(x)
        return s

    def local_search(s, start_time, max_time=25):
        best_set = set(s)
        best_sum = sum(Fraction(1, x) for x in best_set)
        all_candidates = set(range(1, 257))
        
        while time.time() < start_time + max_time:
            improved = False
            
            # 1. Add: try to add smallest available numbers first
            to_add = sorted(all_candidates - best_set)
            for x in to_add:
                if time.time() >= start_time + max_time: break
                if not has_3ap_with_x(best_set, x):
                    best_set.add(x)
                    best_sum += Fraction(1, x)
                    improved = True
            if improved:
                continue
                
            # 2. Swap: remove a larger number, add a smaller one if possible
            to_remove = sorted(best_set, reverse=True)
            for rem in to_remove:
                if time.time() >= start_time + max_time: break
                
                # Find best x to add such that x < rem and valid
                best_add = None
                best_gain = Fraction(0)
                
                # Try smaller numbers first (higher reciprocal)
                for x in range(1, rem):
                    if time.time() >= start_time + max_time: break
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
                    break
            
            # 3. Double swap: remove two, add two
            if not improved and len(best_set) >= 2:
                to_remove = sorted(best_set, reverse=True)
                found = False
                for i in range(len(to_remove)):
                    if time.time() >= start_time + max_time: break
                    for j in range(i+1, len(to_remove)):
                        if time.time() >= start_time + max_time: break
                        rem1, rem2 = to_remove[i], to_remove[j]
                        temp_set = best_set - {rem1, rem2}
                        
                        # Try to add two smaller numbers
                        best_add1 = None
                        best_add2 = None
                        best_gain = Fraction(0)
                        
                        for x1 in range(1, min(rem1, rem2)):
                            if time.time() >= start_time + max_time: break
                            if x1 in temp_set:
                                continue
                            if not has_3ap_with_x(temp_set, x1):
                                temp_set_with_x1 = temp_set | {x1}
                                for x2 in range(1, min(rem1, rem2)):
                                    if x2 == x1 or x2 in temp_set_with_x1:
                                        continue
                                    if not has_3ap_with_x(temp_set_with_x1, x2):
                                        gain = Fraction(1, x1) + Fraction(1, x2) - Fraction(1, rem1) - Fraction(1, rem2)
                                        if gain > best_gain:
                                            best_gain = gain
                                            best_add1 = x1
                                            best_add2 = x2
                        
                        if best_add1 is not None and best_add2 is not None:
                            best_set.remove(rem1)
                            best_set.remove(rem2)
                            best_set.add(best_add1)
                            best_set.add(best_add2)
                            best_sum += best_gain
                            improved = True
                            found = True
                            break
                    if found:
                        break
            
            if not improved:
                break
                
        return best_set, best_sum

    best_overall_set = set()
    best_overall_sum = Fraction(0)
    
    # Try multiple construction strategies
    strategies = [
        greedy_construct_priority,
        greedy_construct_descending,
        greedy_construct_mixed,
    ]
    
    for strategy in strategies:
        if time.time() >= deadline - 10:
            break
        s0 = strategy()
        s1, sum1 = local_search(s0, time.time(), max_time=15)
        if sum1 > best_overall_sum:
            best_overall_set = s1
            best_overall_sum = sum1
    
    # If time remains, try random restarts with kicks
    while time.time() < deadline - 8:
        s_rand = greedy_construct_priority()
        if len(s_rand) > 3:
            items = list(s_rand)
            num_remove = random.randint(2, 5)
            for _ in range(num_remove):
                if items:
                    idx = random.randint(0, len(items) - 1)
                    s_rand.remove(items[idx])
                    items.pop(idx)
        
        s3, sum3 = local_search(s_rand, time.time(), max_time=12)
        if sum3 > best_overall_sum:
            best_overall_set = s3
            best_overall_sum = sum3
            
    return sorted(best_overall_set)
