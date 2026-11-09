"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
    N = 256
    
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.sqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    # Collect primes
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Precompute prime factorization for all numbers 2..256
    factorization = [None] * (N + 1)
    for n in range(2, N + 1):
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
            p += 1
        if temp > 1:
            factors.append((temp, 1))
        factorization[n] = factors
    
    def can_divide(a, b, c):
        """Check if a divides b*c using prime factorizations."""
        b_factors = dict(factorization[b])
        c_factors = dict(factorization[c])
        for p, exp_a in factorization[a]:
            exp_bc = b_factors.get(p, 0) + c_factors.get(p, 0)
            if exp_bc < exp_a:
                return False
        return True
    
    def can_add_to_set(x, S):
        """Check if x can be added to set S without violating constraints.
        For all distinct a,b in S: a does NOT divide x*b, and x does NOT divide a*b."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if j == i:
                    continue
                b = S_list[j]
                # Check a | x*b
                if can_divide(a, x, b):
                    return False
                # Check x | a*b
                if can_divide(x, a, b):
                    return False
        return True
    
    def is_valid_set(s):
        """Check if set s satisfies the constraint for all triples."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                b = s_list[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = s_list[k]
                    if can_divide(a, b, c):
                        return False
        return True
    
    # Start with all primes
    current_set = set(primes)
    best_set = set(primes)
    
    # Try to add composites one by one (greedy)
    composites = [i for i in range(2, N + 1) if not is_prime[i]]
    
    # Sort composites by complexity (fewer distinct prime factors first, then by value)
    def complexity(x):
        f = factorization[x]
        return (len(f), x)
    
    composites.sort(key=complexity)
    
    # Multiple greedy passes with different orderings
    best_greedy = set(primes)
    
    # Pass 1: sort by number of prime factors, then value
    current_set = set(primes)
    for c in composites:
        if time.time() > deadline - 20:
            break
        if can_add_to_set(c, current_set):
            current_set.add(c)
    
    if len(current_set) > len(best_greedy):
        best_greedy = set(current_set)
    
    # Pass 2: sort by value descending (try larger composites first)
    current_set = set(primes)
    composites_desc = sorted(composites, reverse=True)
    for c in composites_desc:
        if time.time() > deadline - 20:
            break
        if can_add_to_set(c, current_set):
            current_set.add(c)
    
    if len(current_set) > len(best_greedy):
        best_greedy = set(current_set)
    
    # Pass 3: random orderings
    import random
    rng = random.Random(42)
    for trial in range(10):
        if time.time() > deadline - 15:
            break
        current_set = set(primes)
        shuffled = list(composites)
        rng.shuffle(shuffled)
        for c in shuffled:
            if time.time() > deadline - 15:
                break
            if can_add_to_set(c, current_set):
                current_set.add(c)
        
        if len(current_set) > len(best_greedy):
            best_greedy = set(current_set)
    
    best_set = best_greedy
    
    # Local search: try swapping elements to enable adding more
    all_numbers = list(range(2, N + 1))
    improved = True
    max_iters = 5
    iter_count = 0
    while improved and time.time() < deadline - 8 and iter_count < max_iters:
        improved = False
        iter_count += 1
        current_list = sorted(best_set)
        for a in current_list:
            if time.time() > deadline - 8:
                break
            for b in all_numbers:
                if b in best_set:
                    continue
                # Try swapping a with b
                new_set = set(best_set)
                new_set.discard(a)
                new_set.add(b)
                # Check if b can be added to (best_set - {a})
                if can_add_to_set(b, new_set - {b}):
                    # Try to greedily add more elements
                    temp_set = set(new_set)
                    for c in composites:
                        if c in temp_set:
                            continue
                        if time.time() > deadline - 8:
                            break
                        if can_add_to_set(c, temp_set):
                            temp_set.add(c)
                    if len(temp_set) > len(best_set):
                        best_set = temp_set
                        improved = True
                        break
            if improved:
                break
    
    # Try: remove one element, then greedily add back multiple
    for a in list(best_set):
        if time.time() > deadline - 4:
            break
        reduced_set = set(best_set)
        reduced_set.discard(a)
        # Try to add composites back
        temp_set = set(reduced_set)
        for c in composites:
            if c == a:
                continue
            if time.time() > deadline - 4:
                break
            if can_add_to_set(c, temp_set):
                temp_set.add(c)
        if len(temp_set) > len(best_set):
            best_set = temp_set
    
    # Try: remove two elements, then greedily add back
    best_set_list = sorted(best_set)
    for i in range(len(best_set_list)):
        if time.time() > deadline - 2:
            break
        for j in range(i+1, len(best_set_list)):
            if time.time() > deadline - 2:
                break
            a = best_set_list[i]
            b = best_set_list[j]
            reduced_set = set(best_set)
            reduced_set.discard(a)
            reduced_set.discard(b)
            # Try to add composites back
            temp_set = set(reduced_set)
            for c in composites:
                if c == a or c == b:
                    continue
                if time.time() > deadline - 2:
                    break
                if can_add_to_set(c, temp_set):
                    temp_set.add(c)
            if len(temp_set) > len(best_set):
                best_set = temp_set
                break
    
    # Final verification
    if not is_valid_set(best_set):
        # Fallback to primes
        best_set = set(primes)
    
    result = sorted(best_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
