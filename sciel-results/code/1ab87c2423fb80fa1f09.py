import time
import random
from math import isqrt

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    candidates = list(range(2, P['n'] + 1))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    def num_distinct_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    composites.sort(key=num_distinct_prime_factors)
    
    best_set = []
    best_size = 0

    def can_add(products, x):
        for p in products:
            if p * x in products:
                return False
        return True

    def get_products(S):
        prods = {1}
        for x in S:
            new_prods = {p * x for p in prods}
            prods.update(new_prods)
        return prods

    # Target sizes to try in descending order
    target_sizes = [18, 17, 16, 15, 14, 13, 12]
    
    found = False
    for target in target_sizes:
        if found:
            break
        if time.time() > deadline - 5.0:
            break
            
        # Start with all primes
        # If target < len(primes), we need to remove some primes.
        # For now, assume target >= len(primes) (12).
        # If target < 12, we just pick the first 'target' primes?
        # No, we want max size. So we start with 12 primes.
        
        base_set = primes[:]
        base_prods = get_products(base_set)
        base_size = len(base_set)
        
        if base_size > target:
            # We need to remove primes.
            # This is a different search.
            # For simplicity, if target < 12, we just return the first 'target' primes.
            # But 12 is already a valid set.
            # So we only care about target >= 12.
            best_set = base_set[:target]
            best_size = target
            found = True
            break
            
        # We need to add (target - base_size) composites
        needed = target - base_size
        
        # DFS to add composites
        def dfs_add(current_set, current_prods, current_size, composites_list):
            if current_size == target:
                return current_set[:]
            
            if not composites_list:
                return None
                
            if current_size + len(composites_list) < target:
                return None
                
            for i in range(len(composites_list)):
                if time.time() > deadline - 2.0:
                    return None
                    
                x = composites_list[i]
                
                if can_add(current_prods, x):
                    new_set = current_set + [x]
                    new_prods = set(current_prods)
                    for p in current_prods:
                        new_prods.add(p * x)
                    
                    result = dfs_add(new_set, new_prods, current_size + 1, composites_list[i+1:])
                    if result:
                        return result
                        
            return None

        # Try deterministic order
        result = dfs_add(base_set, base_prods, base_size, composites)
        if result:
            best_set = result
            best_size = target
            found = True
            break
        else:
            # Try randomized orders
            rng = random.Random(42 + target)
            # Limit number of restarts based on time
            max_restarts = 200
            for _ in range(max_restarts):
                if time.time() > deadline - 5.0:
                    break
                shuffled_composites = composites[:]
                rng.shuffle(shuffled_composites)
                result = dfs_add(base_set, base_prods, base_size, shuffled_composites)
                if result:
                    best_set = result
                    best_size = target
                    found = True
                    break

    # Fallback greedy if DFS failed to find anything (or if we ran out of time)
    if best_size < 12: # Should not happen as primes are valid
        # Just return primes
        best_set = primes
        best_size = len(primes)
    elif best_size < 18 and time.time() < deadline - 2.0:
        # Try a quick greedy to see if we can beat the DFS result (unlikely if DFS is good, but safe)
        def try_greedy(order):
            current = []
            prods = {1}
            for v in order:
                valid = True
                new_prods = []
                for p in prods:
                    np = v * p
                    if np in prods:
                        valid = False
                        break
                    new_prods.append(np)
                if valid:
                    current.append(v)
                    prods.update(new_prods)
            return current

        rng = random.Random(99)
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            sol = try_greedy(shuffled)
            if len(sol) > best_size:
                best_size = len(sol)
                best_set = sol[:]

    return [int(x) for x in best_set]
