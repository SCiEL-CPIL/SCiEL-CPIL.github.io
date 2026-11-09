import math
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    MAX_SUM = 256

    # Precompute coprime matrix
    coprime = [[False] * (MAX_SUM + 1) for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        for j in range(i + 1, MAX_SUM + 1):
            if gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True

    def is_pairwise_coprime_fast(sums_list):
        distinct = list(set(sums_list))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if not coprime[distinct[i]][distinct[j]]:
                    return False
        return True

    # Precompute primes up to 256
    is_prime = [False] * (MAX_SUM + 1)
    for n in range(2, MAX_SUM + 1):
        is_prime[n] = True
    for n in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[n]:
            for m in range(n*n, MAX_SUM + 1, n):
                is_prime[m] = False

    # Verify seed
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    seed_sums = [a + b for a in seed_A for b in seed_B]
    if is_pairwise_coprime_fast(seed_sums):
        best_A = seed_A
        best_B = seed_B
        best_count = len(set(seed_sums))
    else:
        best_A = [1, 2, 3]
        best_B = [1, 2, 3]
        best_count = 0

    # Phase 1: Systematic search for 9 distinct prime sums
    # The 3x3 grid is: sum[i][j] = c + da[i] + db[j]
    # where da = [0, da1, da2], db = [0, db1, db2]
    # All 9 values must be prime for guaranteed pairwise coprimality

    # Enumerate offset patterns with small offsets first
    max_off = 50

    found_9_prime = False
    for da1 in range(1, max_off + 1):
        if time.time() > deadline - 20:
            break
        for da2 in range(da1 + 1, max_off + 1):
            if time.time() > deadline - 20:
                break
            if da2 + 1 > 127:  # a + da2 <= 128, a >= 1
                continue
            for db1 in range(1, max_off + 1):
                if time.time() > deadline - 20:
                    break
                for db2 in range(db1 + 1, max_off + 1):
                    if time.time() > deadline - 20:
                        break
                    if db2 + 1 > 127:  # b + db2 <= 128, b >= 1
                        continue

                    da_vals = [0, da1, da2]
                    db_vals = [0, db1, db2]
                    c_min = 2
                    c_max = 256 - da2 - db2
                    if c_max < c_min:
                        continue

                    # Quick filter: check if any offset combination would force even numbers
                    # If c is even, then c + odd offset gives odd, c + even offset gives even
                    # For all 9 to be prime, we need careful parity management

                    for c in range(c_min, c_max + 1):
                        ok = True
                        for dv in da_vals:
                            for ev in db_vals:
                                val = c + dv + ev
                                if val < 2 or val > MAX_SUM or not is_prime[val]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue

                        # All 9 are prime! Check feasibility
                        a_min = max(1, c - 128 + db2)
                        a_max = min(128 - da2, c - 1)

                        if a_min <= a_max:
                            a = a_min
                            b = c - a
                            A = [a, a + da1, a + da2]
                            B = [b, b + db1, b + db2]

                            if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                                best_A = A
                                best_B = B
                                best_count = 9
                                found_9_prime = True
                                return [A, B]

    # Phase 2: Randomized search for 9 distinct pairwise-coprime sums (not necessarily prime)
    if time.time() < deadline - 5 and best_count < 9:
        random.seed(42)
        attempts = 0
        while time.time() < deadline - 5 and attempts < 200000:
            attempts += 1
            if attempts % 10000 == 0 and time.time() > deadline - 10:
                break

            # Random offsets
            da1 = random.randint(1, 60)
            da2 = random.randint(da1 + 1, 70)
            db1 = random.randint(1, 60)
            db2 = random.randint(db1 + 1, 70)

            if da2 > 127 or db2 > 127:
                continue

            da_vals = [0, da1, da2]
            db_vals = [0, db1, db2]
            c_min = 2
            c_max = 256 - da2 - db2
            if c_max < c_min:
                continue

            c = random.randint(c_min, c_max)

            sums9 = [c + dv + ev for dv in da_vals for ev in db_vals]
            distinct = list(set(sums9))

            if len(distinct) <= best_count:
                continue

            if not is_pairwise_coprime_fast(sums9):
                continue

            a_min = max(1, c - 128 + db2)
            a_max = min(128 - da2, c - 1)

            if a_min <= a_max:
                a = a_min
                b = c - a
                A = [a, a + da1, a + da2]
                B = [b, b + db1, b + db2]

                if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                    all_sums = [x + y for x in A for y in B]
                    ds = list(set(all_sums))
                    if len(ds) > best_count and is_pairwise_coprime_fast(ds):
                        best_A = A
                        best_B = B
                        best_count = len(ds)
                        if best_count == 9:
                            return [A, B]

    return [best_A, best_B]
