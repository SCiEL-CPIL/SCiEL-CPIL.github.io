import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _is_prime(n):
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

def _num_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count

def _conflicts_with_set(x, S):
    """Check if adding x to S creates any bad triple with a pair in S."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if _is_bad_triple(x, a, b):
                return True
    return False

def _greedy_with_order(order):
    S = set()
    for x in order:
        if x not in S and not _conflicts_with_set(x, S):
            S.add(x)
    return S

def _local_search(S, deadline):
    """
    Improve S by:
      1. Adding any element that doesn't conflict (grows size).
      2. 1-swaps: remove a, add x (keeps size, changes structure).
    Returns the best set found.
    """
    S = set(S)
    full_set = set(range(1, 129))
    best_size = len(S)
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline:
        improved = False

        # Phase 1: try to add any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflicts_with_set(x, S):
                S.add(x)
                best_size = max(best_size, len(S))
                improved = True
                break

        if improved:
            no_improve_count = 0
            continue

        # Phase 2: try 1-swaps
        found_swap = False
        S_list = list(S)
        random.shuffle(S_list)
        for a in S_list:
            if time.time() > deadline:
                break
            S.remove(a)
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflicts_with_set(x, S):
                    S.add(x)
                    found_swap = True
                    break
            if not found_swap:
                S.add(a)

        if found_swap:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break

    return S

def _validate(S):
    """Return True if S has no bad triples."""
    S_list = sorted(S)
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin for 90 s limit

    all_elements = list(range(1, 129))

    # Build deterministic orderings
    orderings = []

    # 1. Primes first
    primes = [x for x in all_elements if _is_prime(x)]
    non_primes = [x for x in all_elements if not _is_prime(x)]
    orderings.append(primes + non_primes)

    # 2. Fewer divisors first
    orderings.append(sorted(all_elements, key=lambda x: _num_divisors(x)))

    # 3. Reverse (largest first)
    orderings.append(list(reversed(all_elements)))

    # 4. Forward (smallest first)
    orderings.append(all_elements)

    # 5. Powers of 2 first, then others
    powers_of_2 = [x for x in all_elements if x & (x - 1) == 0]
    others = [x for x in all_elements if x not in powers_of_2]
    orderings.append(powers_of_2 + others)

    # Random orderings
    random.seed(42)
    for _ in range(8):
        orderings.append(random.sample(all_elements, 128))

    best_set = set()
    best_size = 0

    # Run all greedy passes
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_with_order(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Continue with more random restarts if time permits
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        candidates = random.sample(all_elements, 128)
        S = _greedy_with_order(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Local search on the best set found
    if time.time() < deadline - 2.0:
        improved = _local_search(best_set, deadline - 2.0)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)

    # Final validation: ensure no bad triples exist
    if not _validate(best_set):
        # Fallback: rebuild with a simple valid set
        best_set = {1}

    if not best_set:
        best_set = {1}

    return list(best_set)
