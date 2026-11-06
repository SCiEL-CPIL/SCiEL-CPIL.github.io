import time
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    primes = _sieve(1000)
    all_primes = primes
    
    best_L = 0
    best_classes = []
    
    def try_greedy_fast(prime_order, deadline):
        """Fast greedy: use boolean array for coverage tracking."""
        nonlocal best_L, best_classes
        
        used_primes = set()
        assigned = {}  # p -> r
        budget = Fraction(1)
        
        # Coverage array for 1..100000
        covered = bytearray(100001)
        
        for n in range(1, 100001):
            if time.time() > deadline:
                return
            
            if covered[n]:
                if n > best_L:
                    best_L = n
                    if n % 500 == 0:
                        best_classes = [[p, r] for p, r in assigned.items()]
                continue
            
            # Need to cover n - find best prime
            best_p = None
            best_score = -1
            for p in prime_order:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                r = n % p
                # Score: how many new numbers in [1, n+1000] would this cover?
                # Approximate by counting uncovered in a window
                new_cov = 0
                for k in range(n, min(n + 200, 100001), p):
                    if not covered[k]:
                        new_cov += 1
                # Prefer primes that cover more, break ties by smaller prime
                score = new_cov * 1000 - p
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                L = n - 1
                if L > best_L:
                    best_L = L
                    best_classes = [[p, r] for p, r in assigned.items()]
                return
            
            r = n % best_p
            used_primes.add(best_p)
            assigned[best_p] = r
            budget -= Fraction(1, best_p)
            
            # Mark all numbers congruent to r mod best_p as covered
            for k in range(r if r >= 1 else best_p, 100001, best_p):
                covered[k] = 1
            
            if n > best_L:
                best_L = n
                if n % 500 == 0:
                    best_classes = [[p, r] for p, r in assigned.items()]
        
        if best_L < 100000:
            best_L = min(100000, best_L)
        best_classes = [[p, r] for p, r in assigned.items()]
        return
    
    def local_search(classes, deadline):
        """Try to improve by swapping or adding classes."""
        nonlocal best_L, best_classes
        
        if not classes:
            return
        
        classes = [list(c) for c in classes]
        used_primes = set(p for p, r in classes)
        budget = Fraction(1) - sum(Fraction(1, p) for p, r in classes)
        
        # Compute current coverage
        covered = bytearray(100001)
        for p, r in classes:
            for k in range(r if r >= 1 else p, 100001, p):
                covered[k] = 1
        
        # Find first gap
        gap_pos = -1
        for n in range(1, 100001):
            if not covered[n]:
                gap_pos = n
                break
        
        if gap_pos == -1:
            if 100000 > best_L:
                best_L = 100000
                best_classes = [list(c) for c in classes]
            return
        
        # Try to add a new prime to cover gap_pos
        for p in all_primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget < cost:
                continue
            r = gap_pos % p
            # Check how many new numbers this covers
            new_cov = 0
            for k in range(r if r >= 1 else p, 100001, p):
                if not covered[k]:
                    new_cov += 1
            if new_cov > 0:
                # Try adding this class
                new_classes = classes + [[p, r]]
                new_used = used_primes | {p}
                new_budget = budget - cost
                new_covered = bytearray(covered)
                for k in range(r if r >= 1 else p, 100001, p):
                    new_covered[k] = 1
                
                # Find new gap
                new_gap = -1
                for n in range(1, 100001):
                    if not new_covered[n]:
                        new_gap = n
                        break
                
                if new_gap > gap_pos or new_gap == -1:
                    # Improvement!
                    classes = new_classes
                    used_primes = new_used
                    budget = new_budget
                    covered = new_covered
                    break
        
        # Compute final L
        L = 0
        for n in range(1, 100001):
            if covered[n]:
                L = n
            else:
                break
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        return
    
    # Strategy 1: Greedy with increasing primes
    try_greedy_fast(all_primes, deadline)
    
    # Strategy 2: Greedy with decreasing primes
    try_greedy_fast(list(reversed(all_primes)), deadline)
    
    # Strategy 3: Greedy with medium-first order
    medium_primes = sorted(all_primes, key=lambda p: abs(p - 500))
    try_greedy_fast(medium_primes, deadline)
    
    # Strategy 4: Try excluding small primes
    for exclude in [[2], [2,3], [2,3,5], [2,3,5,7]]:
        if time.time() > deadline - 5:
            break
        filtered = [p for p in all_primes if p not in exclude]
        try_greedy_fast(filtered, deadline)
    
    # Local search on best result
    if time.time() < deadline - 3 and best_classes:
        local_search(best_classes, deadline)
    
    # Ensure valid result
    if not best_classes:
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
    
    return best_classes
