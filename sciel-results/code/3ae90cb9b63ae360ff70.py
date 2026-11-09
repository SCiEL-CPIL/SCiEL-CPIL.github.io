import time
import random

# EVOLVE-BLOCK-START
P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute divisibility: divs[a] = set of b in 2..n such that a | b
    # Actually we need: for a given a, which pairs (b,c) satisfy a | b*c
    # Key insight: a | b*c iff for every prime p^k || a, the combined exponent in b*c >= k
    
    # Precompute prime factorizations
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
    
    # Precompute factorizations for all numbers
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
    
    def is_valid_set(S):
        """Check if set S is valid: no three distinct a,b,c with a|b*c"""
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(len(S_list)):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(len(S_list)):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if divides_product(a, b, c):
                        return False
        return True
    
    def can_add(S, new_elem):
        """Check if adding new_elem to set S keeps it valid"""
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(len(S_list)):
                if j == i:
                    continue
                b = S_list[j]
                # Check a | new_elem * b
                if divides_product(a, new_elem, b):
                    return False
                # Check a | b * new_elem (same as above)
                # Check new_elem | a * b
                if divides_product(new_elem, a, b):
                    return False
        return True
    
    # Greedy construction with multiple orderings
    best_set = []
    
    # Strategy 1: Sort by number of prime factors (ascending) - primes first
    sorted_by_pf = sorted(candidates, key=lambda x: sum(factorizations[x].values()))
    
    selected = set()
    for a in sorted_by_pf:
        if time.time() > deadline - 10:
            break
        if can_add(selected, a):
            selected.add(a)
    
    best_set = list(selected)
    
    # Strategy 2: Sort by value descending (large numbers are "safer")
    selected = set()
    for a in sorted(candidates, reverse=True):
        if time.time() > deadline - 10:
            break
        if can_add(selected, a):
            selected.add(a)
    
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 3: Sort by conflict count
    # Count for each a: how many pairs (b,c) in candidates with b<c, b!=a, c!=a have a|b*c
    conflict_count = {}
    for a in candidates:
        cnt = 0
        for b in candidates:
            if b == a:
                continue
            for c in candidates:
                if c == a or c <= b:
                    continue
                if divides_product(a, b, c):
                    cnt += 1
        conflict_count[a] = cnt
    
    sorted_by_conflict = sorted(candidates, key=lambda x: conflict_count[x])
    
    selected = set()
    for a in sorted_by_conflict:
        if time.time() > deadline - 10:
            break
        if can_add(selected, a):
            selected.add(a)
    
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Local improvement: try to swap elements to increase set size
    # For each element in best_set, try removing it and adding a different element
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_set_set = set(best_set)
        for i, elem in enumerate(best_set):
            if time.time() > deadline - 5:
                break
            # Try removing elem and adding a candidate not in set
            best_set_set.remove(elem)
            for candidate in candidates:
                if candidate in best_set_set:
                    continue
                if can_add(best_set_set, candidate):
                    best_set_set.add(candidate)
                    improved = True
                    break
            if not improved:
                best_set_set.add(elem)  # restore
    
    best_set = list(best_set_set)
    
    # Verify the result
    for i, a in enumerate(best_set):
        for j, b in enumerate(best_set):
            if i == j:
                continue
            for k, c in enumerate(best_set):
                if i == k or j == k:
                    continue
                if divides_product(a, b, c):
                    # Invalid! Fallback to primes
                    is_prime = [True] * (n + 1)
                    is_prime[0] = False
                    if n >= 1:
                        is_prime[1] = False
                    for p in range(2, int(n**0.5) + 1):
                        if is_prime[p]:
                            for q in range(p*p, n + 1, p):
                                is_prime[q] = False
                    return [x for x in range(2, n + 1) if is_prime[x]]
    
    return best_set
# EVOLVE-BLOCK-END
