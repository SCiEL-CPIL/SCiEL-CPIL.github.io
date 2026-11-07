import random
import time
from itertools import combinations

def construct():
    all_nums = list(range(2, 41))
    
    def is_valid(S):
        """
        Check if all 2^k subset products of S are distinct using meet-in-the-middle.
        """
        k = len(S)
        if k == 0:
            return True
        if k == 1:
            return True
        if k == 2:
            return S[0] != S[1]
        
        # Split into two halves
        mid = k // 2
        A = list(S[:mid])
        B = list(S[mid:])
        
        # Compute all subset products of A
        prods_A = {1}
        for x in A:
            new_prods = {p * x for p in prods_A}
            prods_A.update(new_prods)
        
        # Compute all subset products of B
        prods_B = {1}
        for x in B:
            new_prods = {p * x for p in prods_B}
            prods_B.update(new_prods)
        
        # Check for collisions: any a in prods_A and b in prods_B where a*b == some other product?
        # Actually, we need to check if any non-trivial combination gives the same product.
        # The full set of products is {a * b for a in prods_A, b in prods_B}
        # We need all 2^k of these to be distinct.
        # This is equivalent to: the map (a, b) -> a*b is injective over prods_A x prods_B.
        # A collision occurs if a1*b1 = a2*b2 with (a1,b1) != (a2,b2).
        # This means a1/a2 = b2/b1, i.e., there exist distinct pairs.
        
        # Simpler: just compute all products and check count
        all_prods = set()
        for a in prods_A:
            for b in prods_B:
                all_prods.add(a * b)
        
        return len(all_prods) == (1 << k)
    
    def can_add(S, x):
        """Check if adding x to S keeps all subset products distinct."""
        return is_valid(list(S) + [x])
    
    # Primes up to 40 - these are always multiplicatively independent
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Start with all primes
    current_set = set(primes)
    best_set = set(primes)
    best_size = len(primes)
    
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    # Precompute which composites might be useful
    composites = [x for x in all_nums if x not in primes]
    
    # Heuristic ordering: try smaller composites first? Or random?
    # Let's use a priority based on "prime density" - composites with fewer prime factors
    # might be easier to add
    
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Sort composites by number of prime factors (fewer = potentially easier to add)
    composites_sorted = sorted(composites, key=lambda x: (prime_factor_count(x), x))
    
    stuck_count = 0
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        candidates = [x for x in all_nums if x not in current_set]
        
        # Try in order of prime factor count first, then random
        for x in composites_sorted:
            if x in current_set:
                continue
            if time.time() > deadline:
                break
            if can_add(current_set, x):
                current_set.add(x)
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                stuck_count = 0
                break
        
        if not added:
            # Try random candidates too
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(current_set, x):
                    current_set.add(x)
                    added = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    stuck_count = 0
                    break
        
        if not added:
            stuck_count += 1
            
            # 2. Try swaps to escape local optimum
            if current_set and candidates:
                swap_found = False
                # Try multiple random swaps
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    y = random.choice(list(current_set))
                    x = random.choice(candidates)
                    if x == y:
                        continue
                    new_set = (current_set - {y}) | {x}
                    if is_valid(list(new_set)):
                        current_set = new_set
                        swap_found = True
                        stuck_count = 0
                        break
                
                if not swap_found and stuck_count > 50:
                    # Restart from best set
                    current_set = set(best_set)
                    stuck_count = 0
    
    return list(best_set)
