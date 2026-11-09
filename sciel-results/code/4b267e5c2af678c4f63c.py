"""Rewritten program for large integer sets with no two-other-product divisibility."""
import time
import random
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    all_candidates = list(range(2, 257))
    n_total = len(all_candidates)
    idx = {x: i for i, x in enumerate(all_candidates)}

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

    factor_cache = {x: prime_factors(x) for x in all_candidates}

    def can_add_fast(candidate, current_list):
        """Check if candidate can be added to current_list."""
        n = len(current_list)
        cand_factors = factor_cache[candidate]
        
        for i in range(n):
            bi = current_list[i]
            fi = factor_cache[bi]
            for j in range(i + 1, n):
                bj = current_list[j]
                fj = factor_cache[bj]
                divisible = True
                for p, exp in cand_factors.items():
                    if fi.get(p, 0) + fj.get(p, 0) < exp:
                        divisible = False
                        break
                if divisible:
                    return False
        
        for i in range(n):
            a = current_list[i]
            fa = factor_cache[a]
            for j in range(n):
                if j == i:
                    continue
                x = current_list[j]
                fx = factor_cache[x]
                divisible = True
                for p, exp in fa.items():
                    if cand_factors.get(p, 0) + fx.get(p, 0) < exp:
                        divisible = False
                        break
                if divisible:
                    return False
        return True

    def is_valid_set(s):
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

    best_set = set()
    best_size = 0

    def num_prime_factors(n):
        return sum(factor_cache[n].values())

    # Multiple orderings
    orderings = [
        sorted(primes) + sorted(composites, reverse=True),
        sorted(all_candidates, reverse=True),
        sorted(all_candidates),
        sorted(primes, reverse=True) + sorted(composites, reverse=True),
        [x for x in sorted(primes, reverse=True) if x > 100] + 
        [x for x in sorted(primes, reverse=True) if x <= 100] + 
        sorted(composites, reverse=True),
        sorted(all_candidates, key=lambda x: (num_prime_factors(x), -x)),
        sorted(all_candidates, key=lambda x: (-num_prime_factors(x), x)),
        sorted(all_candidates, key=lambda x: (num_prime_factors(x), x)),
    ]

    random.seed(42)
    for _ in range(40):
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for ordering in orderings:
        if time.time() > deadline - 20:
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

    # Greedy fill
    if time.time() < deadline - 20:
        best_list = list(best_set)
        not_in_best = [x for x in all_candidates if x not in best_set]
        improved = True
        while improved and time.time() < deadline - 18:
            improved = False
            still_not_in = []
            for c in not_in_best:
                if time.time() > deadline - 15:
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

    # 1-for-1 swaps
    if time.time() < deadline - 25:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            best_list = list(best_set)
            improved = False
            for i in range(len(best_list)):
                if time.time() > deadline - 12:
                    break
                elem = best_list[i]
                test_set = set(best_set) - {elem}
                test_list = [x for x in best_list if x != elem]
                for c in all_candidates:
                    if c in test_set:
                        continue
                    if can_add_fast(c, test_list):
                        best_set = test_set | {c}
                        best_list = test_list + [c]
                        improved = True
                        break
                if time.time() > deadline - 12:
                    break
            if not improved:
                break

    # 1-for-2 swaps
    if time.time() < deadline - 20:
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 15:
                break
            elem = best_list[i]
            test_set = set(best_set) - {elem}
            test_list = [x for x in best_list if x != elem]
            added = []
            for c in all_candidates:
                if c in test_set:
                    continue
                if len(added) >= 2:
                    break
                if can_add_fast(c, test_list + added):
                    added.append(c)
            if len(added) >= 2:
                new_set = test_set | set(added)
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)

    # 2-for-2 swaps
    if time.time() < deadline - 20:
        best_list = list(best_set)
        n_best = len(best_list)
        for i in range(n_best):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, n_best):
                if time.time() > deadline - 15:
                    break
                test_set = set(best_set) - {best_list[i], best_list[j]}
                test_list = [x for x in best_list if x != best_list[i] and x != best_list[j]]
                added = []
                for c in all_candidates:
                    if c in test_set:
                        continue
                    if len(added) >= 2:
                        break
                    if can_add_fast(c, test_list + added):
                        added.append(c)
                if len(added) >= 2:
                    new_set = test_set | set(added)
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        best_size = len(best_set)

    # Final fill
    if time.time() < deadline - 10:
        best_list = list(best_set)
        not_in_best = [x for x in all_candidates if x not in best_set]
        improved = True
        while improved and time.time() < deadline - 8:
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

    # Final swap rounds
    if time.time() < deadline - 8:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            best_list = list(best_set)
            improved = False
            for i in range(len(best_list)):
                if time.time() > deadline - 3:
                    break
                elem = best_list[i]
                test_set = set(best_set) - {elem}
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

    result = sorted(best_set)
    if not is_valid_set(result):
        current = set()
        current_list = []
        for c in all_candidates:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        result = sorted(current)

    return [int(x) for x in result]

# EVOLVE-BLOCK-END
