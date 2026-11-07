import random
import time
import sys

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Max possible sum is 256 * 30 = 7680
    MAX_SUM = 7680
    
    # Precompute divisibility relationships
    # multiples[i] contains all multiples of i up to MAX_SUM
    # This allows O(1) check: does a divide b? -> b in multiples[a]
    multiples = [set() for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1, i):
            multiples[i].add(j)
            
    def check_divisibility_conflict(a, b):
        """Check if a and b have a divisibility conflict (a|b or b|a)."""
        if a == b:
            return False # Distinct values required, but if values are same, it's a collision not divisibility
        if b in multiples[a]:
            return True
        if a in multiples[b]:
            return True
        return False

    def try_add_element(current_elements, current_sums, x):
        """
        Try to add x to the current set.
        Returns (new_sums, valid)
        """
        # 1. x itself must not be in current_sums
        if x in current_sums:
            return None, False
        
        # 2. Generate new sums
        new_sums_set = set()
        for s in current_sums:
            ns = s + x
            # Collision check: ns must not be in current_sums
            if ns in current_sums:
                return None, False
            # Collision check: ns must not be duplicate within new_sums
            # Since elements are distinct and sums were distinct, s+x are distinct?
            # If s1 + x == s2 + x, then s1 == s2. Since sums are distinct, new sums are distinct.
            new_sums_set.add(ns)
        
        new_sums_set.add(x)
        
        # 3. Divisibility Checks
        # Check between new_sums and current_sums
        for ns in new_sums_set:
            for s in current_sums:
                if check_divisibility_conflict(ns, s):
                    return None, False
        
        # Check within new_sums
        # Optimization: sort new_sums to check smaller vs larger
        new_sums_list = sorted(new_sums_set)
        for i in range(len(new_sums_list)):
            a = new_sums_list[i]
            for j in range(i + 1, len(new_sums_list)):
                b = new_sums_list[j]
                # a < b, so only check if a divides b
                if b in multiples[a]:
                    return None, False
                    
        return new_sums_set, True

    best_size = 0
    best_set = [31, 32, 34] # Initial valid set
    
    # Verify initial set
    if len(best_set) > 0:
        # Simple validation for initial set to be safe
        sums = set()
        n = len(best_set)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += best_set[i]
            if s in sums:
                best_set = []
                break
            sums.add(s)
        if best_set:
            sl = sorted(sums)
            valid = True
            for i in range(len(sl)):
                for j in range(i+1, len(sl)):
                    if sl[j] % sl[i] == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_size = len(best_set)
            else:
                best_set = []
                best_size = 0

    random.seed(42)
    
    # Strategy: Greedy construction with random restarts
    # We try to build a set from scratch by adding elements one by one.
    
    num_restarts = 2000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random starting point
        # Start with a single element
        start_val = random.randint(1, 256)
        current_elements = [start_val]
        current_sums = {start_val}
        
        # Candidate pool: all numbers 1..256 not in current_elements
        # We want to try adding elements in a specific order.
        # Heuristic: Larger elements might be better to avoid small divisors?
        # Or smaller elements to keep sums small?
        # Let's try a mix. We'll shuffle candidates.
        
        candidates = [x for x in range(1, 257) if x != start_val]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current_elements) >= 30:
                break
                
            new_sums, valid = try_add_element(current_elements, current_sums, x)
            if valid:
                current_elements.append(x)
                current_sums = new_sums
                if len(current_elements) > best_size:
                    best_size = len(current_elements)
                    best_set = current_elements[:]
                
        # Optional: Try to improve the current set by swapping? 
        # For now, just restart.
        
    return list(best_set)
