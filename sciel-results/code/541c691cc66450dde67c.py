"""Combined approach: Prime-factor conflict detection + First-Improvement + SA + multi-swap."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    N = 512
    
    # Precompute values and prime factors for all numbers 1..511
    values = [Fraction(0)] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N
    
    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)
        
    # Identify primes
    primes = [a for a in range(1, N) if is_prime_arr[a]]
    
    # Initialize with all primes
    selected = set(primes)
    
    # Build mapping: prime -> set of selected elements that have this prime as a factor
    prime_to_selected = {}
    for b in selected:
        for p in prime_factors[b]:
            if p not in prime_to_selected:
                prime_to_selected[p] = set()
            prime_to_selected[p].add(b)
    
    def apply_move(add_candidates, remove_elements):
        """Apply a move: remove elements, add candidates. Update prime_to_selected."""
        for b in remove_elements:
            for p in prime_factors[b]:
                if p in prime_to_selected:
                    prime_to_selected[p].discard(b)
                    if not prime_to_selected[p]:
                        del prime_to_selected[p]
        for c in add_candidates:
            for p in prime_factors[c]:
                if p not in prime_to_selected:
                    prime_to_selected[p] = set()
                prime_to_selected[p].add(c)
        selected.difference_update(remove_elements)
        selected.update(add_candidates)
    
    def get_score():
        return sum(values[a] for a in selected)
    
    current_score = get_score()
    
    # Phase 1: 1-for-k First-Improvement Search
    # Scan candidates and apply the first move that improves the score.
    improved = True
    while improved:
        if time.time() > deadline - 40:
            break
            
        improved = False
        for c in range(1, N):
            if c in selected:
                continue
            
            # Find conflicts using prime factor mapping
            conflict_set = set()
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
            
            conflict_cost = sum(values[b] for b in conflict_set)
            gain = values[c] - conflict_cost
            
            if gain > 0:
                apply_move([c], list(conflict_set))
                current_score += gain
                improved = True
                break  # Restart scan
    
    # Phase 2: 2-for-k Best-Improvement Search
    if time.time() < deadline - 25:
        # Sort composites by value descending
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:80]
        
        improved = True
        while improved:
            if time.time() > deadline - 15:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    
                    # Find conflicts
                    conflict_set = set()
                    for p in prime_factors[c1]:
                        if p in prime_to_selected:
                            conflict_set.update(prime_to_selected[p])
                    for p in prime_factors[c2]:
                        if p in prime_to_selected:
                            conflict_set.update(prime_to_selected[p])
                    
                    conflict_cost = sum(values[b] for b in conflict_set)
                    gain = values[c1] + values[c2] - conflict_cost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(conflict_set)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Phase 3: 3-for-k Best-Improvement Search
    if time.time() < deadline - 15:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:50]
        
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    for k in range(j + 1, len(top_composites)):
                        c3 = top_composites[k]
                        if c3 in selected:
                            continue
                        if gcd(c1, c3) != 1 or gcd(c2, c3) != 1:
                            continue
                        
                        # Find conflicts
                        conflict_set = set()
                        for c in (c1, c2, c3):
                            for p in prime_factors[c]:
                                if p in prime_to_selected:
                                    conflict_set.update(prime_to_selected[p])
                        
                        conflict_cost = sum(values[b] for b in conflict_set)
                        gain = values[c1] + values[c2] + values[c3] - conflict_cost
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_add = [c1, c2, c3]
                            best_remove = list(conflict_set)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Phase 4: Simulated Annealing (if time permits)
    if time.time() < deadline - 5:
        # SA Parameters
        T = 0.05
        T_min = 0.001
        cooling = 0.995
        
        sa_deadline = deadline - 2
        
        while T > T_min and time.time() < sa_deadline:
            # Pick a random element to remove
            if not selected:
                break
            a_out = random.choice(list(selected))
            
            # Pick a random element to add from unselected
            unselected = [a for a in range(1, N) if a not in selected]
            if not unselected:
                break
            a_in = random.choice(unselected)
            
            # Check if they can coexist (coprime check with all selected except a_out)
            valid = True
            for b in selected:
                if b == a_out:
                    continue
                if gcd(a_in, b) != 1:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
            # Compute score change
            old_score = current_score
            apply_move([a_in], [a_out])
            new_score = get_score()
            delta = new_score - old_score
            
            if delta > 0:
                # Accept
                current_score = new_score
            else:
                # Accept with probability exp(delta / T)
                import math
                prob = math.exp(float(delta) / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    apply_move([a_out], [a_in])
            
            T *= cooling
    
    # Final pass of 1-for-k to clean up
    if time.time() < deadline - 5:
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for c in range(1, N):
                if c in selected:
                    continue
                conflict_set = set()
                for p in prime_factors[c]:
                    if p in prime_to_selected:
                        conflict_set.update(prime_to_selected[p])
                conflict_cost = sum(values[b] for b in conflict_set)
                gain = values[c] - conflict_cost
                if gain > best_gain:
                    best_gain = gain
                    best_add = [c]
                    best_remove = list(conflict_set)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Final Verification
    result = list(selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")
    
    return result
# EVOLVE-BLOCK-END
