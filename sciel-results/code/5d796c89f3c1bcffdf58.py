"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute prime factorizations for all numbers
    def prime_factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    factorizations = {}
    for i in range(2, n + 1):
        factorizations[i] = prime_factorize(i)
    
    def divides_product(a, b, c):
        """Check if a divides b*c using factorizations"""
        for p, exp_a in factorizations[a].items():
            exp_bc = factorizations[b].get(p, 0) + factorizations[c].get(p, 0)
            if exp_bc < exp_a:
                return False
        return True
        
    def can_add(S, new_elem):
        """
        Check if adding new_elem to set S keeps it valid.
        S is a set of ints. new_elem is not in S.
        We need to check all triples (a, b, c) involving new_elem are valid.
        Let a, b be distinct elements in S.
        Triples: (new_elem, a, b), (a, new_elem, b), (a, b, new_elem).
        1. new_elem does not divide a * b
        2. a does not divide new_elem * b
        3. b does not divide new_elem * a (symmetric to 2)
        """
        S_list = list(S)
        m = len(S_list)
        # Check condition 1: new_elem | a*b
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if divides_product(new_elem, a, b):
                    return False
        
        # Check condition 2 & 3: a | new_elem*b or b | new_elem*a
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if i == j:
                    continue
                b = S_list[j]
                # Check a | new_elem * b
                if divides_product(a, new_elem, b):
                    return False
        return True

    def is_valid_set(S):
        """Check if set S is valid: no three distinct a,b,c with a|b*c"""
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(m):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if divides_product(a, b, c):
                        return False
        return True

    def conflict_count(S, elem):
        """Count how many existing elements in S would conflict with elem if added."""
        count = 0
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if divides_product(elem, a, b):
                    count += 1
            for j in range(m):
                if i == j:
                    continue
                b = S_list[j]
                if divides_product(a, elem, b):
                    count += 1
        return count

    best_set = set()
    
    # Strategy: Multiple Randomized Greedy with different ordering strategies
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        current_set = set()
        
        # Different ordering strategies for diversity
        if r % 4 == 0:
            # Pure random
            shuffled_candidates = candidates.copy()
            random.shuffle(shuffled_candidates)
        elif r % 4 == 1:
            # Prefer numbers with fewer prime factors (less likely to divide products)
            shuffled_candidates = sorted(candidates, key=lambda x: (len(factorizations[x]), random.random()))
        elif r % 4 == 2:
            # Prefer larger numbers
            shuffled_candidates = sorted(candidates, key=lambda x: (-x, random.random()))
        else:
            # Prefer numbers that are products of large primes
            shuffled_candidates = sorted(candidates, key=lambda x: (-max(factorizations[x].keys()) if factorizations[x] else 0, random.random()))
        
        for a in shuffled_candidates:
            if time.time() > deadline - 10:
                break
            if can_add(current_set, a):
                current_set.add(a)
                
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Local Improvement: Try to swap elements to increase set size
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_set_list = list(best_set)
        
        # Try removing one element and adding two others
        for i in range(len(best_set_list)):
            if time.time() > deadline - 5:
                break
            u = best_set_list[i]
            trial_set = best_set - {u}
            
            # Try to add any two elements from candidates not in trial_set
            added_count = 0
            added_elements = []
            
            for v in candidates:
                if v in trial_set:
                    continue
                if can_add(trial_set, v):
                    trial_set.add(v)
                    added_elements.append(v)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count >= 2:
                # This swap increases the set size by 1
                best_set = trial_set
                improved = True
                break
            
            # Also try single swap (same size but might enable further additions)
            # For now, skip this as it doesn't directly increase size
    
    # Alternative: Run more restarts if time permits
    while time.time() < deadline - 5:
        # Try a few more restarts with different strategies
        current_set = set()
        
        # Mix of strategies
        strategy = random.choice(['random', 'few_factors', 'large', 'large_primes'])
        
        if strategy == 'random':
            shuffled_candidates = candidates.copy()
            random.shuffle(shuffled_candidates)
        elif strategy == 'few_factors':
            shuffled_candidates = sorted(candidates, key=lambda x: (len(factorizations[x]), random.random()))
        elif strategy == 'large':
            shuffled_candidates = sorted(candidates, key=lambda x: (-x, random.random()))
        else:
            shuffled_candidates = sorted(candidates, key=lambda x: (-max(factorizations[x].keys()) if factorizations[x] else 0, random.random()))
        
        for a in shuffled_candidates:
            if time.time() > deadline - 5:
                break
            if can_add(current_set, a):
                current_set.add(a)
                
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Final Verification
    if not is_valid_set(best_set):
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Ensure all elements are Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
