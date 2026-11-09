"""Rewritten program for large integer sets with no two-other-product divisibility."""
import time
import random
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Candidate set: all integers from 2 to 256
    all_candidates = list(range(2, 257))
    n_total = len(all_candidates)

    # Precompute prime factorizations for faster divisibility checks
    def prime_factors(n):
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

    # Precompute factorizations for all candidates
    factor_cache = {x: prime_factors(x) for x in all_candidates}

    def can_add_fast(candidate, current_list):
        """Fast check: can we add candidate to current_list?
        Condition: for all distinct a,b,c in {candidate} ∪ current_list, a ∤ b*c.
        This means:
        1. candidate ∤ b*c for all distinct b,c in current_list
        2. for each a in current_list: a ∤ candidate * x for all x in current_list, x != a
        """
        n = len(current_list)
        cand_factors = factor_cache[candidate]
        
        # Check 1: candidate does not divide b*c for any distinct b,c in current_list
        # b*c is divisible by candidate iff for all primes p in cand_factors, 
        # exponent in b*c >= exponent in candidate
        for i in range(n):
            bi = current_list[i]
            fi = factor_cache[bi]
            for j in range(i + 1, n):
                bj = current_list[j]
                fj = factor_cache[bj]
                # Check if candidate divides bi * bj
                divisible = True
                for p, exp in cand_factors.items():
                    total_exp = fi.get(p, 0) + fj.get(p, 0)
                    if total_exp < exp:
                        divisible = False
                        break
                if divisible:
                    return False
        
        # Check 2: for each a in current_list, a does not divide candidate * x for any x != a
        for i in range(n):
            a = current_list[i]
            fa = factor_cache[a]
            for j in range(n):
                if j == i:
                    continue
                x = current_list[j]
                fx = factor_cache[x]
                # Check if a divides candidate * x
                divisible = True
                for p, exp in fa.items():
                    total_exp = cand_factors.get(p, 0) + fx.get(p, 0)
                    if total_exp < exp:
                        divisible = False
                        break
                if divisible:
                    return False
        return True

    def is_valid_set(s):
        """Check if set s satisfies the condition."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Greedy approach with multiple orderings
    best_set = set()
    best_size = 0

    # Strategy: prioritize numbers with fewer "conflict partners"
    # A number x has a conflict with y if x divides y*z for some z, or y divides x*z for some z
    # Simpler heuristic: prioritize primes (they have fewer multiplicative relationships)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [x for x in all_candidates if is_prime(x)]
    composites = [x for x in all_candidates if not is_prime(x)]

    # Try different orderings
    orderings = [
        primes + sorted(composites, reverse=True),  # primes first, then large composites
        sorted(all_candidates, reverse=True),  # descending
        sorted(all_candidates),  # ascending
        list(all_candidates),  # natural order
        # Primes in descending order first
        sorted(primes, reverse=True) + sorted(composites, reverse=True),
        # Large primes first
        [x for x in sorted(primes, reverse=True) if x > 100] + 
        [x for x in sorted(primes, reverse=True) if x <= 100] + 
        sorted(composites, reverse=True),
    ]

    # Add more random orderings for diversity
    random.seed(42)
    for _ in range(30):
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Add orderings that prioritize numbers with fewest prime factors
    def num_prime_factors(n):
        return sum(factor_cache[n].values())
    
    orderings.append(sorted(all_candidates, key=lambda x: (num_prime_factors(x), -x)))
    orderings.append(sorted(all_candidates, key=lambda x: (-num_prime_factors(x), x)))

    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        current = set()
        current_list = []
        for c in ordering:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)

    # Local search: try to add elements (greedy fill)
    not_in_best = [x for x in all_candidates if x not in best_set]
    best_list = list(best_set)
    improved = True
    while improved:
        improved = False
        still_not_in = []
        for c in not_in_best:
            if time.time() > deadline - 10:
                still_not_in.append(c)
                continue
            if can_add_fast(c, best_list):
                best_set.add(c)
                best_list.append(c)
                improved = True
                best_size = len(best_set)
            else:
                still_not_in.append(c)
        not_in_best = still_not_in
        if time.time() > deadline - 10:
            break

    # Local search: 1-for-1 swap (remove one, try to add a different one)
    # Multiple rounds of swaps for better convergence
    if time.time() < deadline - 15:
        for swap_round in range(3):
            if time.time() > deadline - 10:
                break
            best_list = list(best_set)
            improved_this_round = False
            for i in range(len(best_list)):
                if time.time() > deadline - 10:
                    break
                elem_to_remove = best_list[i]
                test_set = set(best_set)
                test_set.discard(elem_to_remove)
                test_list = [x for x in best_list if x != elem_to_remove]
                # Try each candidate not in test_set
                for c in all_candidates:
                    if c in test_set:
                        continue
                    if can_add_fast(c, test_list):
                        # Found a valid swap
                        best_set = test_set | {c}
                        best_list = test_list + [c]
                        improved_this_round = True
                        break
                if time.time() > deadline - 10:
                    break
            if not improved_this_round:
                break

    # Local search: 1-for-2 swap (remove one, try to add two others)
    # Also try 2-for-2 swaps
    if time.time() < deadline - 10:
        # 1-for-2 swaps
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            elem_to_remove = best_list[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            test_list = [x for x in best_list if x != elem_to_remove]
            candidates_to_add = [x for x in all_candidates if x not in test_set]
            added = []
            for c in candidates_to_add:
                if time.time() > deadline - 10:
                    break
                if len(added) >= 2:
                    break
                if can_add_fast(c, test_list + added):
                    added.append(c)
            if len(added) >= 2:
                new_set = test_set | set(added)
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)
                    best_list = list(best_set)
        
        # 2-for-2 swaps: remove two, try to add two different ones
        if time.time() < deadline - 10:
            best_list = list(best_set)
            n_best = len(best_list)
            for i in range(n_best):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, n_best):
                    if time.time() > deadline - 10:
                        break
                    test_set = set(best_set)
                    test_set.discard(best_list[i])
                    test_set.discard(best_list[j])
                    test_list = [x for x in best_list if x != best_list[i] and x != best_list[j]]
                    candidates_to_add = [x for x in all_candidates if x not in test_set]
                    added = []
                    for c in candidates_to_add:
                        if time.time() > deadline - 10:
                            break
                        if len(added) >= 2:
                            break
                        if can_add_fast(c, test_list + added):
                            added.append(c)
                    if len(added) >= 2:
                        new_set = test_set | set(added)
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_size = len(best_set)
                            best_list = list(best_set)

    # Second round of greedy fill after swaps
    if time.time() < deadline - 10:
        best_list = list(best_set)
        not_in_best = [x for x in all_candidates if x not in best_set]
        improved = True
        while improved:
            improved = False
            still_not_in = []
            for c in not_in_best:
                if time.time() > deadline - 5:
                    still_not_in.append(c)
                    continue
                if can_add_fast(c, best_list):
                    best_set.add(c)
                    best_list.append(c)
                    improved = True
                    best_size = len(best_set)
                else:
                    still_not_in.append(c)
            not_in_best = still_not_in
            if time.time() > deadline - 5:
                break
    
    # Final aggressive local search: try all possible single-element replacements
    if time.time() < deadline - 5:
        best_list = list(best_set)
        for _ in range(2):  # Two rounds
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(len(best_list)):
                if time.time() > deadline - 3:
                    break
                elem = best_list[i]
                test_set = set(best_set)
                test_set.discard(elem)
                test_list = [x for x in best_list if x != elem]
                for c in all_candidates:
                    if c in test_set:
                        continue
                    if can_add_fast(c, test_list):
                        best_set = test_set | {c}
                        best_list = test_list + [c]
                        improved = True
                        break
                if time.time() > deadline - 3:
                    break
            if not improved:
                break

    # Final verification
    result = sorted(best_set)
    if not is_valid_set(result):
        # Fallback: try to build a valid set from scratch
        current = set()
        current_list = []
        for c in all_candidates:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        result = sorted(current)

    # Ensure we return a list of Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
