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
    deadline = start_time + 145
    
    primes = _sieve(1000)
    all_primes = primes
    
    best_L = 0
    best_classes = []
    
    def compute_L(covered):
        L = 0
        for n in range(1, 100001):
            if covered[n]:
                L = n
            else:
                break
        return L
    
    def greedy_with_lookahead(prime_order, deadline, look_window=500):
        nonlocal best_L, best_classes
        
        used_primes = set()
        assigned = {}
        budget = Fraction(1)
        covered = bytearray(100001)
        
        for n in range(1, 100001):
            if time.time() > deadline:
                break
            
            if covered[n]:
                continue
            
            best_p = None
            best_score = -1
            
            for p in prime_order:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                
                # Count how many numbers in [n, n+look_window] this would cover
                new_cov = 0
                for k in range(n, min(n + look_window, 100001), p):
                    if not covered[k]:
                        new_cov += 1
                
                # Also count total coverage in [1, 100000]
                total_new = 0
                for k in range(r if r >= 1 else p, 100001, p):
                    if not covered[k]:
                        total_new += 1
                
                # Score: prioritize covering many new numbers, prefer smaller primes for tie-break
                score = total_new * 100 + new_cov * 10 - p
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
            
            r = n % best_p
            used_primes.add(best_p)
            assigned[best_p] = r
            budget -= Fraction(1, best_p)
            
            for k in range(r if r >= 1 else best_p, 100001, best_p):
                covered[k] = 1
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [[p, r] for p, r in assigned.items()]
        return best_classes
    
    def repair_and_extend(classes, deadline):
        """Iteratively find gaps and try to fill them by adding or swapping classes."""
        nonlocal best_L, best_classes
        
        if not classes:
            return
        
        classes = [list(c) for c in classes]
        used_primes = set(p for p, r in classes)
        budget = Fraction(1) - sum(Fraction(1, p) for p, r in classes)
        
        covered = bytearray(100001)
        for p, r in classes:
            for k in range(r if r >= 1 else p, 100001, p):
                covered[k] = 1
        
        iterations = 0
        while time.time() < deadline - 2:
            iterations += 1
            if iterations > 100:
                break
            
            # Find first gap
            gap_pos = -1
            for n in range(1, 100001):
                if not covered[n]:
                    gap_pos = n
                    break
            
            if gap_pos == -1:
                L = 100000
                if L > best_L:
                    best_L = L
                    best_classes = [list(c) for c in classes]
                break
            
            # Try to add a new prime to cover gap_pos
            added = False
            for p in all_primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = gap_pos % p
                new_covered = bytearray(covered)
                for k in range(r if r >= 1 else p, 100001, p):
                    new_covered[k] = 1
                
                new_L = compute_L(new_covered)
                if new_L > gap_pos:
                    # This addition helps
                    classes.append([p, r])
                    used_primes.add(p)
                    budget -= cost
                    covered = new_covered
                    added = True
                    break
            
            if not added:
                # Try swapping: remove one existing class, add a different one
                swapped = False
                for i, (p_old, r_old) in enumerate(classes):
                    cost_old = Fraction(1, p_old)
                    for p_new in all_primes:
                        if p_new in used_primes or p_new == p_old:
                            continue
                        cost_new = Fraction(1, p_new)
                        # Check if we can afford the swap
                        if budget + cost_old < cost_new:
                            continue
                        
                        r_new = gap_pos % p_new
                        # Simulate the swap
                        temp_covered = bytearray(100001)
                        for j, (pj, rj) in enumerate(classes):
                            if j == i:
                                continue
                            for k in range(rj if rj >= 1 else pj, 100001, pj):
                                temp_covered[k] = 1
                        for k in range(r_new if r_new >= 1 else p_new, 100001, p_new):
                            temp_covered[k] = 1
                        
                        new_L = compute_L(temp_covered)
                        if new_L > gap_pos:
                            # Perform swap
                            classes[i] = [p_new, r_new]
                            used_primes.discard(p_old)
                            used_primes.add(p_new)
                            budget += cost_old - cost_new
                            covered = temp_covered
                            swapped = True
                            break
                    if swapped:
                        break
            
            if not added and not swapped:
                # Can't improve further
                L = compute_L(covered)
                if L > best_L:
                    best_L = L
                    best_classes = [list(c) for c in classes]
                break
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        return best_classes
    
    # Strategy 1: Greedy with increasing primes
    greedy_with_lookahead(all_primes, deadline)
    
    # Strategy 2: Greedy with decreasing primes
    greedy_with_lookahead(list(reversed(all_primes)), deadline)
    
    # Strategy 3: Greedy with medium-first order
    medium_primes = sorted(all_primes, key=lambda p: abs(p - 500))
    greedy_with_lookahead(medium_primes, deadline)
    
    # Strategy 4: Try various exclusions
    for exclude in [[2], [2,3], [2,3,5], [2,3,5,7], [3], [3,5]]:
        if time.time() > deadline - 10:
            break
        filtered = [p for p in all_primes if p not in exclude]
        greedy_with_lookahead(filtered, deadline)
    
    # Repair and extend on best result
    if time.time() < deadline - 5 and best_classes:
        repair_and_extend(best_classes, deadline)
    
    # Final validation
    if not best_classes:
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
    
    # Ensure validity
    used = set()
    total = Fraction(0)
    for p, r in best_classes:
        if p in used or p > 1000 or r < 0 or r >= p:
            # Invalid, fallback
            primes_13_plus = [p for p in all_primes if p >= 13]
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
            break
        used.add(p)
        total += Fraction(1, p)
    
    if total > 1:
        # Trim to fit budget
        best_classes = []
        used = set()
        total = Fraction(0)
        for p, r in sorted(best_classes, key=lambda x: -x[0]):
            if p not in used:
                cost = Fraction(1, p)
                if total + cost <= 1:
                    best_classes.append([p, r])
                    used.add(p)
                    total += cost
    
    return best_classes
