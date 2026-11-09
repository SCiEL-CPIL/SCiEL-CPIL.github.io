"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    # Generate powerful numbers up to L
    # A powerful number n has all prime exponents >= 2
    # Every powerful number can be written as a^2 * b^3 where a,b >= 1
    L = 5 * 10**8  # Target limit

    # Generate all a^2 * b^3 <= L
    powerful_set = set()
    powerful_set.add(1)

    max_b = int(round(L ** (1.0/3.0))) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
            elif a * a * b3 > L:
                break

    powerful = sorted(powerful_set)
    n_p = len(powerful)

    # If we ran out of time generating, use what we have
    if time.time() > deadline - 60.0:
        pass  # proceed with whatever we have

    # Strategy: Count all pair sums to find N with most representations
    # But O(n^2) may be too slow for large n_p
    # Alternative: For a fixed N, count pairs using two pointers in O(n_p)
    # Then search over candidate N values

    # First, let's do a full O(n^2) count if n_p is small enough
    # n_p ~ 10^4 for L=10^8, so n_p^2 ~ 10^8 which is borderline
    # For L=5*10^8, n_p might be ~2*10^4, n_p^2 ~ 4*10^8 which is too slow in Python

    # Better approach: Use the fact that we want to maximize count of pairs (a,b) with a<=b, a+b=N
    # This is equivalent to finding the mode of the sumset

    # Let's use a hybrid: 
    # 1. If n_p is small enough (< 15000), do full O(n^2) enumeration
    # 2. Otherwise, use two-pointer counting for candidate N values

    if n_p <= 15000:
        # Full enumeration
        count = defaultdict(int)
        for i in range(n_p):
            a = powerful[i]
            for j in range(i, n_p):
                b = powerful[j]
                N = a + b
                if N > 10**18:
                    break
                count[N] += 1
                if time.time() > deadline - 20.0:
                    break
            if time.time() > deadline - 20.0:
                break

        if count:
            best_N = max(count, key=count.get)
            best_count = count[best_N]
            # Collect pairs
            pairs = []
            pset = set(powerful)
            for a in powerful:
                b = best_N - a
                if b >= a and b in pset:
                    pairs.append([a, b])
            return {'N': best_N, 'pairs': pairs}

    # For larger n_p, use two-pointer counting for candidate N values
    # Candidate N values: sample across the range [powerful[0]+powerful[1], powerful[-1]+powerful[-2]]
    min_sum = powerful[0] + powerful[1]
    max_sum = powerful[-1] + powerful[-2]

    def count_pairs_for_N(N):
        """Count pairs (a,b) with a<=b, a+b=N, both powerful, using two pointers."""
        i = 0
        j = n_p - 1
        count = 0
        pairs = []
        while i <= j:
            s = powerful[i] + powerful[j]
            if s == N:
                count += 1
                pairs.append([powerful[i], powerful[j]])
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        return count, pairs

    # Coarse search: sample ~500 candidate N values
    best_count = 0
    best_N = min_sum
    best_pairs = []

    n_candidates = 500
    for k in range(n_candidates):
        if time.time() > deadline - 30.0:
            break
        if k == n_candidates - 1:
            N = max_sum
        else:
            N = int(min_sum + (max_sum - min_sum) * k / (n_candidates - 1))
        if N < 2 or N > 10**18:
            continue
        cnt, pairs = count_pairs_for_N(N)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs

    # Fine search: around best_N, check nearby values
    fine_range = max(100, best_count * 20)
    for offset in range(-fine_range, fine_range + 1):
        if time.time() > deadline - 10.0:
            break
        N = best_N + offset
        if N < 2 or N > 10**18:
            continue
        cnt, pairs = count_pairs_for_N(N)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs

    # Second fine search if we found something good
    if best_count > 5:
        fine_range2 = max(200, best_count * 30)
        for offset in range(-fine_range2, fine_range2 + 1):
            if time.time() > deadline - 5.0:
                break
            N = best_N + offset
            if N < 2 or N > 10**18:
                continue
            cnt, pairs = count_pairs_for_N(N)
            if cnt > best_count:
                best_count = cnt
                best_N = N
                best_pairs = pairs

    if not best_pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
