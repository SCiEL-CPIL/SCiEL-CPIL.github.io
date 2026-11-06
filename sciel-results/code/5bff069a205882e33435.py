import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin
    
    def can_add(x, S, used_sums):
        """
        Check if x can be added to S.
        S is a sorted list of selected numbers.
        used_sums is a set of all triple sums from S.
        Returns (valid, new_sums) where new_sums is the set of sums involving x.
        """
        new_sums = set()
        
        # Triple (x, x, x)
        s = 3 * x
        if s in used_sums:
            return False, set()
        new_sums.add(s)
        
        # Triples (x, x, y) and (x, y, y) and (x, y, z)
        # We must consider all y, z in S.
        # Since S is sorted, we can just iterate.
        # Note: The triple is unordered. The sum is commutative.
        # We just need to generate all sums a+b+c where {a,b,c} is a multiset 
        # formed by taking x and two elements from S (or one from S and x twice, etc).
        
        # Case 1: Two x's and one y from S
        for y in S:
            s = 2 * x + y
            if s in used_sums or s in new_sums:
                return False, set()
            new_sums.add(s)
            
        # Case 2: One x and two y's from S
        # Subcase 2a: y == z
        for y in S:
            s = x + 2 * y
            if s in used_sums or s in new_sums:
                return False, set()
            new_sums.add(s)
                
        # Subcase 2b: y < z
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                s = x + y + z
                if s in used_sums or s in new_sums:
                    return False, set()
                new_sums.add(s)
                
        return True, new_sums

    def run_greedy(order, time_budget, start_set=None):
        S = list(start_set) if start_set else []
        S.sort()
        used_sums = set()
        
        # Precompute sums for initial set
        if S:
            for a, b, c in combinations_with_replacement(S, 3):
                used_sums.add(a + b + c)
                
        for x in order:
            if time.time() - start_time > time_budget:
                break
            if x in S:
                continue
            valid, new_sums = can_add(x, S, used_sums)
            if valid:
                S.append(x)
                S.sort()
                used_sums.update(new_sums)
                
        return S

    best_set = []
    
    # Strategy 1: Standard Greedy (Ascending)
    if time.time() - start_time < time_limit - 10:
        s = run_greedy(range(1, 4097), time_limit - (time.time() - start_time))
        if len(s) > len(best_set):
            best_set = s
            
    # Strategy 2: Greedy (Descending)
    if time.time() - start_time < time_limit - 10:
        s = run_greedy(range(4096, 0, -1), time_limit - (time.time() - start_time))
        if len(s) > len(best_set):
            best_set = s
            
    # Strategy 3: Randomized Greedy
    random.seed(42)
    for trial in range(1000):
        if time.time() - start_time > time_limit - 5:
            break
        order = list(range(1, 4097))
        random.shuffle(order)
        s = run_greedy(order, time_limit - (time.time() - start_time))
        if len(s) > len(best_set):
            best_set = s
            # If we found a large set, we might want to try to extend it further
            # or just keep it.
            
    # Strategy 4: Local Search / Repair on the best set
    if time.time() - start_time < time_limit - 2 and len(best_set) > 0:
        S = best_set[:]
        S.sort()
        used_sums = set()
        for a, b, c in combinations_with_replacement(S, 3):
            used_sums.add(a + b + c)
            
        # Try to swap elements to allow adding new ones
        # This is complex. A simpler repair:
        # If we are stuck at size k, try to remove one element and add two? No, we want to maximize size.
        # Try to find an element in S that can be replaced by a smaller one to free up sum space?
        
        # Alternative: Try to extend the best set by scanning remaining numbers
        # The greedy already does this, but maybe a different order for the remaining numbers helps?
        
        # Let's try a "sparse" random order
        # Weight larger numbers higher?
        
        for trial in range(500):
            if time.time() - start_time > time_limit - 2:
                break
            # Bias towards numbers that are far from existing ones?
            # Just try random again with a different seed or bias
            order = list(range(1, 4097))
            # Sort by distance to closest element in best_set?
            # This might be slow.
            random.shuffle(order)
            s = run_greedy(order, time_limit - (time.time() - start_time), start_set=best_set)
            if len(s) > len(best_set):
                best_set = s

    # Final Verification
    def verify(s):
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = []
        for a, b, c in combinations_with_replacement(s, 3):
            sums.append(a + b + c)
        return len(sums) == len(set(sums))
    
    if not verify(best_set):
        # Fallback
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
    return best_set
