import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    N = 256
    all_elements = list(range(2, N + 1))
    all_set = set(all_elements)
    
    # Precompute divisor sets for each number in 2..256
    # divisors[x] = set of d in 2..256 such that d | x
    divisors = [set() for _ in range(N + 1)]
    for x in range(2, N + 1):
        for d in range(2, x + 1):
            if x % d == 0:
                divisors[x].add(d)
    
    # Precompute: for each a, the set of (b,c) pairs where a | b*c
    # Actually, let's think differently.
    # Conflict: a | b*c for distinct a,b,c in S.
    # For a fixed a, the "bad" pairs (b,c) are those where a | b*c.
    # We want to avoid having any such triple in S.
    
    # Key insight: For a fixed a, define for each b != a:
    #   a | b*c  <=>  a/gcd(a,b) | c
    # Let a_b = a / gcd(a,b). Then we need c to be a multiple of a_b.
    # So for given a, b in S (b != a), the "forbidden" c values are multiples of a_b.
    
    # Let's precompute for each a and each b: the set of c values (in 2..256, c != a, c != b)
    # such that a | b*c.
    # Actually, we can compute this on the fly efficiently.
    
    def check_validity(S):
        """Check if S is valid: no three distinct a,b,c in S with a | b*c."""
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                g = gcd(a, b)
                a_prime = a // g
                if a_prime == 1:
                    # a | b*c for any c, so we need no other c in S distinct from a,b
                    # If |S| >= 3, this is a conflict
                    if len(S) >= 3:
                        return False
                else:
                    # Need c in S, c != a, c != b, such that a_prime | c
                    for k, c in enumerate(S_list):
                        if k == i or k == j:
                            continue
                        if c % a_prime == 0:
                            return False
        return True
    
    def count_conflicts(S):
        """Count total number of violating triples in S."""
        S_list = list(S)
        count = 0
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                g = gcd(a, b)
                a_prime = a // g
                if a_prime == 1:
                    # a | b*c for any c != a, c != b
                    count += max(0, len(S) - 2)
                else:
                    for k, c in enumerate(S_list):
                        if k == i or k == j:
                            continue
                        if c % a_prime == 0:
                            count += 1
        return count
    
    def can_add(x, S):
        """Check if x can be added to S without creating any new conflicts."""
        # Check: for each a in S, does a | x*c for some c in S, c != a, c != x?
        for a in S:
            g = gcd(a, x)
            a_prime = a // g
            if a_prime == 1:
                # a | x*c for any c, need c in S, c != a
                if len(S) >= 1:
                    return False
            else:
                # Need c in S, c != a, c != x, such that a_prime | c
                for c in S:
                    if c == a:
                        continue
                    if c % a_prime == 0:
                        return False
        
        # Check: for each b in S, does x | b*c for some c in S, c != b, c != x?
        for b in S:
            g = gcd(x, b)
            x_prime = x // g
            if x_prime == 1:
                # x | b*c for any c, need c in S, c != b
                if len(S) >= 1:
                    return False
            else:
                # Need c in S, c != b, c != x, such that x_prime | c
                for c in S:
                    if c == b:
                        continue
                    if c % x_prime == 0:
                        return False
        
        return True
    
    def remove_element(x, S):
        """Remove x from S."""
        S.discard(x)
    
    def add_element(x, S):
        """Add x to S."""
        S.add(x)
    
    best_S = set()
    best_size = 0
    
    # Strategy: Start with a good initial set, then do local search.
    # Good initial sets: primes, or numbers with large smallest prime factor.
    
    # Let's try multiple strategies:
    
    # Strategy 1: Start with all primes
    primes = [x for x in range(2, N + 1) if all(x % p != 0 for p in range(2, int(x**0.5) + 1))]
    S1 = set(primes)
    
    # Strategy 2: Start with numbers whose smallest prime factor > some threshold
    def smallest_prime_factor(x):
        if x % 2 == 0:
            return 2
        for p in range(3, int(x**0.5) + 1, 2):
            if x % p == 0:
                return p
        return x
    
    spf = [0] * (N + 1)
    for x in range(2, N + 1):
        spf[x] = smallest_prime_factor(x)
    
    # Strategy 3: Greedy construction
    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    # Try different orderings
    orders = []
    orders.append(sorted(all_elements, key=lambda x: -spf[x]))  # Large SPF first
    orders.append(sorted(all_elements, key=lambda x: -x))  # Large first
    orders.append(sorted(all_elements, key=lambda x: spf[x]))  # Small SPF first
    orders.append(list(reversed(all_elements)))
    orders.append(sorted(all_elements, key=lambda x: random.random()))
    
    # Run multiple randomized greedy constructions
    for run in range(20):
        if time.time() > deadline:
            break
        
        random.shuffle(orders[-1])
        S = greedy_construct(orders[-1])
        
        # Local search: try to add back removed elements
        removed = list(all_set - S)
        random.shuffle(removed)
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for x in removed[:]:
                if x in S:
                    continue
                if can_add(x, S):
                    S.add(x)
                    removed.remove(x)
                    added_any = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Now do more intensive local search on the best set found
    if time.time() < deadline:
        S = set(best_S)
        removed = list(all_set - S)
        
        # Iterative improvement: try swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any removed element
            for x in removed[:]:
                if time.time() > deadline:
                    break
                if can_add(x, S):
                    S.add(x)
                    removed.remove(x)
                    improved = True
            
            # Try swaps: remove y from S, add x from removed
            for x in removed[:]:
                if time.time() > deadline:
                    break
                for y in list(S):
                    if time.time() > deadline:
                        break
                    # Try: remove y, add x
                    S.remove(y)
                    removed.append(y)
                    if can_add(x, S):
                        S.add(x)
                        removed.remove(x)
                        removed.remove(y)
                        improved = True
                    else:
                        S.add(y)
                        removed.remove(y)
    
    # Final local search with more iterations
    if time.time() < deadline:
        S = set(best_S)
        removed = list(all_set - S)
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Try all additions
            for x in removed[:]:
                if time.time() > deadline:
                    break
                if can_add(x, S):
                    S.add(x)
                    removed.remove(x)
            
            # Try swaps
            for x in removed[:]:
                if time.time() > deadline:
                    break
                for y in list(S):
                    if time.time() > deadline:
                        break
                    S.remove(y)
                    removed.append(y)
                    if can_add(x, S):
                        S.add(x)
                        removed.remove(x)
                        removed.remove(y)
                    else:
                        S.add(y)
                        removed.remove(y)
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
    
    # Final verification
    result = [int(x) for x in best_S]
    return result
