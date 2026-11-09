"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    # Precompute prime factors for all possible sums (2..20000)
    MAX_SUM = 20000
    pf = [set() for _ in range(MAX_SUM + 1)]
    for p in range(2, MAX_SUM + 1):
        if not pf[p]:  # p is prime
            for m in range(p, MAX_SUM + 1, p):
                pf[m].add(p)

    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return len(primes)

    def get_primes_for(arr):
        """Return the set of primes for a given array."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return primes

    # Strategy 1: Build compatibility graph for 7-smooth sums and find clique of size 12
    # A sum s is 7-smooth if all its prime factors are in {2, 3, 5, 7}
    def is_7_smooth(s):
        """Check if s has only prime factors in {2, 3, 5, 7}."""
        for p in [2, 3, 5, 7]:
            while s % p == 0:
                s //= p
        return s == 1

    # Precompute which sums are 7-smooth
    smooth_sum = [False] * (MAX_SUM + 1)
    for s in range(2, MAX_SUM + 1):
        smooth_sum[s] = is_7_smooth(s)

    # Build adjacency: two numbers a, b are compatible if a+b is 7-smooth
    # We'll use bitsets for efficient clique finding
    # Universe: numbers 1..10000
    # For each number, store a bitset of compatible numbers
    # But 10000 numbers is too large for a full bitset approach in Python.
    # Instead, we'll do a smarter search.

    # Key insight: if all numbers have the same parity, all sums are even.
    # If all numbers are congruent mod 3, sums are 0 mod 3 or specific values.
    # Let's try: all numbers even -> sums even. Then need sums to be 7-smooth.
    # Even 7-smooth numbers: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 50, 54, 56, 60, 64, 70, 72, 80, 84, 90, 96, 98, 100, ...

    # Better approach: use the structure. If we pick numbers from an arithmetic progression
    # with step that makes sums smooth, we can guarantee the property.

    # Try: numbers = [2*k for k in 1..12] -> sums = 2*(k1+k2), need k1+k2 to be 7-smooth/2...
    # Actually let's just do a direct search with the compatibility constraint.

    # For efficiency, restrict to a smaller candidate set.
    # Numbers whose sums with many other numbers are 7-smooth.
    # Let's precompute for each number x in 1..10000, how many y in 1..10000 have x+y 7-smooth.

    # Actually, let's be smarter. We know that if x and y are both multiples of some base,
    # their sum is also a multiple. Let's try structured families.

    # Family: all numbers are powers of 2 -> sums are 2^a + 2^b = 2^min(a,b)*(1+2^|a-b|)
    # 1+2^k for k>=1: 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049
    # 3=3, 5=5, 9=3^2, 17=17 (NOT 7-smooth!), so powers of 2 don't work for large exponents.

    # Family: all numbers are multiples of 2: x_i = 2*a_i
    # Sum = 2*(a_i + a_j), need a_i + a_j to be 7-smooth.
    # So we need 12 distinct a_i such that all pairwise sums are 7-smooth.

    # Let's try: a_i from a small set where all pairwise sums are 7-smooth.
    # 7-smooth numbers up to 20000: let's generate them.
    seven_smooth = set()
    for a in range(0, 16):  # 2^a up to 2^15 = 32768
        for b in range(0, 16):
            for c in range(0, 16):
                for d in range(0, 16):
                    v = (2**a) * (3**b) * (5**c) * (7**d)
                    if v <= 20000:
                        seven_smooth.add(v)
    seven_smooth = sorted(seven_smooth)

    # Now find 12 distinct positive integers a_1 < a_2 < ... < a_12
    # such that a_i + a_j is in seven_smooth for all i < j.
    # This is a clique problem in the graph where vertices are positive integers
    # and edges connect i,j if i+j is 7-smooth.

    # For efficiency, limit a_i to reasonable range. If a_i <= 10000, then 2*a_i <= 20000.
    # But we need a_i + a_j <= 20000, so a_i < 20000.
    # Let's restrict a_i to 1..10000 (so x_i = 2*a_i <= 20000... wait, x_i <= 10000 so a_i <= 5000).

    # Let's build the graph for a_i in 1..5000.
    # For each a in 1..5000, neighbors are b in 1..5000 where a+b is 7-smooth.

    # Use a different approach: backtracking with forward checking.
    # Order candidates by degree (most connected first) for faster search.

    MAX_A = 5000  # since x = 2*a <= 10000 means a <= 5000
    # Precompute neighbors
    neighbors = [set() for _ in range(MAX_A + 1)]
    for a in range(1, MAX_A + 1):
        for b in range(a + 1, MAX_A + 1):
            s = a + b
            if s <= MAX_SUM and smooth_sum[s]:
                neighbors[a].add(b)
                neighbors[b].add(a)

    # Degree
    degrees = [len(neighbors[a]) for a in range(MAX_A + 1)]

    # Backtracking search for clique of size 12
    def find_clique(target_size, candidates, current_clique):
        """Find a clique of target_size from candidates, extending current_clique."""
        if len(current_clique) == target_size:
            return list(current_clique)
        if time_exceeded():
            return None
        if len(candidates) < target_size - len(current_clique):
            return None

        # Forward checking: filter candidates that are connected to all in current_clique
        if current_clique:
            filtered = []
            for c in candidates:
                ok = True
                for cc in current_clique:
                    if cc not in neighbors[c]:
                        ok = False
                        break
                if ok:
                    filtered.append(c)
            candidates = filtered
            if len(candidates) < target_size - len(current_clique):
                return None

        # Try candidates in order of degree (descending)
        candidates_sorted = sorted(candidates, key=lambda x: -degrees[x])

        for c in candidates_sorted:
            if time_exceeded():
                return None
            new_clique = current_clique + [c]
            new_candidates = [x for x in candidates_sorted if x > c and x in neighbors[c]]
            result = find_clique(target_size, new_candidates, new_clique)
            if result is not None:
                return result
        return None

    # Try to find a clique of size 12 in the a-space
    # Start with candidates ordered by degree
    all_candidates = [a for a in range(1, MAX_A + 1) if degrees[a] >= 11]
    all_candidates.sort(key=lambda x: -degrees[x])

    best_result = None
    best_count = 999

    # First try the clique approach
    if not time_exceeded():
        clique = find_clique(12, all_candidates[:200], [])
        if clique is not None:
            # Convert back: x_i = 2 * a_i
            result = [2 * a for a in clique]
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                cnt = count_distinct_primes(result)
                best_result = result
                best_count = cnt

    # If clique not found, try other strategies
    if best_result is None:
        # Fallback: powers of 2
        powers_of_2 = [2**i for i in range(12)]
        best_result = list(powers_of_2)
        best_count = count_distinct_primes(best_result)

    # Try to improve with local search
    def local_search(arr, max_iters=30000):
        """Local search: try replacing elements to reduce prime count."""
        arr = list(arr)
        current_count = count_distinct_primes(arr)
        best_arr = list(arr)
        best_cnt = current_count
        rng = random.Random(42)

        for it in range(max_iters):
            if time_exceeded():
                break
            idx = rng.randint(0, 11)
            old_val = arr[idx]
            new_val = rng.randint(1, 10000)
            if new_val == old_val:
                continue
            if new_val in arr and arr.index(new_val) != idx:
                continue

            arr[idx] = new_val
            new_primes = get_primes_for(arr)
            new_count = len(new_primes)

            if new_count < current_count:
                current_count = new_count
                if new_count < best_cnt:
                    best_cnt = new_count
                    best_arr = list(arr)
            else:
                if new_count == current_count and rng.random() < 0.3:
                    pass
                else:
                    arr[idx] = old_val
                    continue

        return best_arr, best_cnt

    if not time_exceeded():
        result_arr, result_cnt = local_search(best_result, max_iters=20000)
        if result_cnt < best_count:
            best_count = result_cnt
            best_result = result_arr

    return best_result
# EVOLVE-BLOCK-END
