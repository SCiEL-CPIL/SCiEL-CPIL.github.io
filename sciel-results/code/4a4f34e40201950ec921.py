"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    from math import isqrt
    import random

    # Factor each number 2..40 into prime-exponent tuple.
    def factor(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    # Build exponent vectors for 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_to_vec = {}
    for n in range(2, 41):
        f = factor(n)
        num_to_vec[n] = tuple(f.get(p, 0) for p in primes)
    
    candidates = list(range(2, 41))
    best_set = []
    best_size = 0

    def can_add(sum_set, vec):
        """
        Check if vec can be added to the set represented by sum_set.
        sum_set contains all subset sums of the current vectors.
        Condition: No two subset sums are equal.
        Adding vec creates new sums {s + vec for s in sum_set}.
        Collision if (s1 + vec) == s2 for some s1, s2 in sum_set.
        i.e., s2 - s1 == vec.
        """
        for s in sum_set:
            # Check if s + vec is already in sum_set
            # Since sum_set is a set of tuples, we can construct the new tuple
            new_s = tuple(a + b for a, b in zip(s, vec))
            if new_s in sum_set:
                return False
        return True

    def expand_greedy(candidates_order):
        current_set = []
        sum_set = {tuple([0]*len(primes))}  # Empty subset sum
        
        for n in candidates_order:
            v = num_to_vec[n]
            if can_add(sum_set, v):
                current_set.append(n)
                # Update sum_set
                new_sums = []
                for s in sum_set:
                    new_s = tuple(a + b for a, b in zip(s, v))
                    new_sums.append(new_s)
                sum_set.update(new_sums)
        
        return current_set

    # Try multiple random orderings to find a larger set
    # Also try a few heuristic orderings (e.g., by number of prime factors, by value)
    
    # Heuristic 1: Random
    for _ in range(100):
        random.shuffle(candidates)
        s = expand_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Heuristic 2: Sort by number of prime factors (ascending) - "simpler" numbers first
    sorted_by_pf = sorted(candidates, key=lambda x: sum(1 for _ in factor(x)))
    s = expand_greedy(sorted_by_pf)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Heuristic 3: Sort by value
    s = expand_greedy(candidates)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Heuristic 4: Sort by value descending
    s = expand_greedy(sorted(candidates, reverse=True))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # Heuristic 5: Primes first
    primes_list = [p for p in candidates if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
    non_primes = [n for n in candidates if n not in primes_list]
    s = expand_greedy(primes_list + non_primes)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    return list(best_set)

# EVOLVE-BLOCK-END
