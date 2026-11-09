import time
import random
from math import gcd

def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def try_greedy(order):
        """
        Greedily build a set from 'order' using incremental sum tracking.
        Returns the selected subset.
        """
        current_set = []
        L = 1
        sums = {0}  # Set of current subset sums (scaled by L)
        
        for x in order:
            new_L = lcm(L, x)
            factor = new_L // L
            new_sums = {s * factor for s in sums}
            w = new_L // x
            
            # Check for collisions: new sums must not intersect old sums
            # New sums are {s + w for s in new_sums}
            # Collision if (s + w) in new_sums for any s in new_sums
            valid = True
            for s in new_sums:
                if (s + w) in new_sums:
                    valid = False
                    break
            
            if valid:
                sums = new_sums | {s + w for s in new_sums}
                L = new_L
                current_set.append(x)
        
        return current_set
    
    def local_search_improve(best, time_budget):
        """
        Try to improve best set by removing one element and greedily adding others.
        """
        if not best or time.time() > deadline - 5.0:
            return best
        
        best_set = set(best)
        improved = False
        
        for x in list(best):
            if time.time() > deadline - 5.0:
                break
            
            # Remove x from best
            candidate = [y for y in best if y != x]
            candidate_set = set(candidate)
            
            # Build sums for candidate
            L = 1
            sums = {0}
            for y in candidate:
                new_L = lcm(L, y)
                factor = new_L // L
                new_sums = {s * factor for s in sums}
                w = new_L // y
                sums = new_sums | {s + w for s in new_sums}
                L = new_L
            
            # Try to add remaining elements
            remaining = [y for y in range(1, 23) if y not in candidate_set]
            random.shuffle(remaining)
            
            current = candidate
            for y in remaining:
                if time.time() > deadline - 5.0:
                    break
                new_L = lcm(L, y)
                factor = new_L // L
                new_sums = {s * factor for s in sums}
                w = new_L // y
                
                valid = True
                for s in new_sums:
                    if (s + w) in new_sums:
                        valid = False
                        break
                
                if valid:
                    sums = new_sums | {s + w for s in new_sums}
                    L = new_L
                    current.append(y)
            
            if len(current) > len(best):
                best = current
                improved = True
        
        return best
    
    best = []
    
    # 1. Deterministic orderings
    base_orders = [
        list(range(1, 23)),       # ascending
        list(range(22, 0, -1)),   # descending
        # Powers of 2 first
        [1, 2, 4, 8, 16] + [x for x in range(1, 23) if x not in [1, 2, 4, 8, 16]],
        # Primes first
        [2, 3, 5, 7, 11, 13, 17, 19] + [x for x in range(1, 23) if x not in [2, 3, 5, 7, 11, 13, 17, 19]],
        # Large numbers first (small reciprocals)
        list(range(22, 0, -1)),
        # Small numbers first (large reciprocals)
        list(range(1, 23)),
        # Mix: alternate small and large
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
    ]
    
    for order in base_orders:
        if time.time() > deadline:
            break
        res = try_greedy(order)
        if len(res) > len(best):
            best = res
    
    # 2. Randomized greedy with bias toward powers of 2
    random.seed(42)
    powers_of_2 = [1, 2, 4, 8, 16]
    
    for _ in range(3000):
        if time.time() > deadline - 10.0:
            break
        
        order = list(range(1, 23))
        random.shuffle(order)
        
        # Sometimes put powers of 2 first
        if random.random() < 0.3:
            rest = [x for x in order if x not in powers_of_2]
            order = powers_of_2 + rest
        
        res = try_greedy(order)
        if len(res) > len(best):
            best = res
    
    # 3. Local search improvement
    if time.time() < deadline - 10.0:
        best = local_search_improve(best, deadline - time.time())
    
    # 4. Final round of random restarts if time permits
    if time.time() < deadline - 5.0:
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            order = list(range(1, 23))
            random.shuffle(order)
            res = try_greedy(order)
            if len(res) > len(best):
                best = res
    
    if best:
        return best
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

