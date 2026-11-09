import math
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 85.0

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pairwise_coprime(lst):
        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    # Precompute coprime matrix
    MAX_SUM = 256
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

    # STRUCTURAL CHANGE: Search over prime-based sum grids.
    # A 3x3 sum grid: s[i][j] = A[i] + B[j]
    # Key property: s[i][j] - s[i][0] = B[j] - B[0] (same for all i)
    # And: s[i][j] - s[0][j] = A[i] - A[0] (same for all j)
    # So the grid is determined by: s[0][0], row_diffs = (d1, d2), col_diffs = (e1, e2)
    # where d1 = A[1]-A[0], d2 = A[2]-A[0], e1 = B[1]-B[0], e2 = B[2]-B[0]
    # s[i][j] = s[0][0] + d_i + e_j where d_0=0,d_1=d1,d_2=d2, e_0=0,e_1=e1,e_2=e2

    # New approach: enumerate all valid "row patterns" (3 pairwise-coprime values with consistent diffs)
    # and "col patterns", then combine them.

    # A row pattern is (base, d1, d2) -> values [base, base+d1, base+d2+d1... wait
    # Actually row i has values: c + d_i + e_j for j=0,1,2
    # The "shape" of a row is determined by (e1, e2) relative to base
    # The "shape" of a column is determined by (d1, d2)

    # Let me think differently. The 9 sums are:
    # c, c+e1, c+e2
    # c+d1, c+d1+e1, c+d1+e2
    # c+d2, c+d2+e1, c+d2+e2
    # where e2 here is actually the full offset for B[2], let me rename:
    # Let offsets_A = [0, da1, da2] where da1 = A[1]-A[0], da2 = A[2]-A[0]
    # Let offsets_B = [0, db1, db2] where db1 = B[1]-B[0], db2 = B[2]-B[0]
    # Sum[i][j] = c + offsets_A[i] + offsets_B[j]

    # All 9 must be in [2, 256] (since min sum is 1+1=2, max is 128+128=256)
    # All distinct sums must be pairwise coprime.

    # Phase 1: Try to find 9 distinct pairwise-coprime sums.
    # Strategy: enumerate (da1, da2, db1, db2) and for each, find c such that
    # all 9 values c + da_i + db_j are pairwise coprime.

    # Precompute: for each (da1, da2, db1, db2), what c values work?
    # This is expensive, so let's be smart.

    # Key insight: if all 9 sums are prime, they're automatically pairwise coprime.
    # So search for c, da1, da2, db1, db2 such that all 9 values are prime.

    # Precompute primes up to 256
    is_prime = [False] * (MAX_SUM + 1)
    for n in range(2, MAX_SUM + 1):
        is_prime[n] = True
    for n in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[n]:
            for m in range(n*n, MAX_SUM + 1, n):
                is_prime[m] = False

    found_9 = False

    # Enumerate offset patterns
    max_off = 40  # maximum offset difference

    for da1 in range(1, max_off + 1):
        if time.time() > deadline - 15:
            break
        for da2 in range(da1 + 1, max_off + 1):
            if time.time() > deadline - 15:
                break
            for db1 in range(1, max_off + 1):
                if time.time() > deadline - 15:
                    break
                for db2 in range(db1 + 1, max_off + 1):
                    if time.time() > deadline - 15:
                        break

                    # The 9 sums are c + da_i + db_j
                    # min sum = c, max sum = c + da2 + db2
                    # Need c >= 2, c + da2 + db2 <= 256
                    # Also need A and B elements in [1,128]:
                    # A = [a, a+da1, a+da2], B = [b, b+db1, b+db2]
                    # a >= 1, a + da2 <= 128 => a <= 128 - da2
                    # b >= 1, b + db2 <= 128 => b <= 128 - db2
                    # c = a + b >= 2, c <= 128 - da2 + 128 - db2 = 256 - da2 - db2

                    # For each c in [2, 256 - da2 - db2]:
                    c_min = 2
                    c_max = 256 - da2 - db2
                    if c_max < c_min:
                        continue

                    # Check if all 9 values c + da_i + db_j are prime
                    # da_vals = [0, da1, da2], db_vals = [0, db1, db2]
                    da_vals = [0, da1, da2]
                    db_vals = [0, db1, db2]

                    # For a given c, the 9 values are c + dv + ev for dv in da_vals, ev in db_vals
                    # We need all to be prime.

                    # Instead of checking each c individually, note that the set of values
                    # is {c + x : x in {dv+ev for dv in da_vals, ev in db_vals}}
                    # The offsets are: 0, db1, db2, da1, da1+db1, da1+db2, da2, da2+db1, da2+db2
                    offsets = sorted(set(dv + ev for dv in da_vals for ev in db_vals))

                    # All 9 values must be prime: c + off for off in offsets (with possible repeats if offsets aren't distinct)
                    # But we need all 9 grid positions to be prime, even if some offsets repeat
                    # Actually if offsets repeat, that means two grid positions have same value, which is fine for coprimality
                    # but we want them distinct for max cardinality.

                    # Let's just check: for c in range, are all 9 grid values prime?
                    # This is 9 prime checks per c. Range is at most 254 values.
                    # Total iterations: ~40^4/4 * 254 * 9 ≈ too many.

                    # Better: for each c, check if c + off is prime for all 9 offsets.
                    # Use the is_prime array for O(1) checks.

                    # But 40^4 / 4 ≈ 640000 patterns, each with up to 254 c values = 160M checks. Too slow.

                    # Need to be smarter. Let's limit the search space.
                    # Only try small offsets first.

                    if da2 + db2 > 60:
                        continue

                    for c in range(c_min, c_max + 1):
                        # Check all 9 values are prime
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

                        # All 9 are prime! Now check feasibility of A, B in [1,128]
                        # c = a + b, a >= 1, b >= 1
                        # a + da2 <= 128 => a <= 128 - da2
                        # b + db2 <= 128 => b <= 128 - db2
                        # a >= 1, b >= 1
                        # b = c - a
                        # 1 <= a <= 128 - da2
                        # 1 <= c - a <= 128 - db2
                        # a >= c - (128 - db2) = c - 128 + db2
                        # a <= min(128 - da2, c - 1)

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
                                found_9 = True
                                return [A, B]

    # Phase 2: If no 9-prime grid found, search for 9 distinct pairwise-coprime (not necessarily prime)
    if time.time() < deadline - 10:
        for da1 in range(1, 35):
            if time.time() > deadline - 10:
                break
            for da2 in range(da1 + 1, 35):
                if time.time() > deadline - 10:
                    break
                for db1 in range(1, 35):
                    if time.time() > deadline - 10:
                        break
                    for db2 in range(db1 + 1, 35):
                        if time.time() > deadline - 10:
                            break
                        if da2 + db2 > 55:
                            continue

                        da_vals = [0, da1, da2]
                        db_vals = [0, db1, db2]
                        c_min = 2
                        c_max = 256 - da2 - db2
                        if c_max < c_min:
                            continue

                        for c in range(c_min, c_max + 1):
                            # Check all 9 values are pairwise coprime
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

    # Phase 3: Broader search for 8 distinct pairwise-coprime sums
    if time.time() < deadline - 5 and best_count < 8:
        # Allow non-strictly-increasing offsets
        for da1 in range(1, 40):
            if time.time() > deadline - 5:
                break
            for da2 in range(1, 40):
                if da1 == da2:
                    continue
                if time.time() > deadline - 5:
                    break
                for db1 in range(1, 40):
                    if time.time() > deadline - 5:
                        break
                    for db2 in range(1, 40):
                        if db1 == db2:
                            continue
                        if time.time() > deadline - 5:
                            break
                        if abs(da1 - da2) + abs(db1 - db2) > 60:
                            continue

                        da_vals = [0, da1, da2]
                        db_vals = [0, db1, db2]
                        c_min = 2
                        c_max = 256 - max(da2, da1) - max(db2, db1)
                        if c_max < c_min:
                            continue

                        for c in range(c_min, c_max + 1):
                            sums9 = [c + dv + ev for dv in da_vals for ev in db_vals]
                            distinct = list(set(sums9))
                            if len(distinct) <= best_count:
                                continue
                            if not is_pairwise_coprime_fast(sums9):
                                continue

                            # Feasibility
                            max_da = max(da1, da2)
                            max_db = max(db1, db2)
                            a_min = max(1, c - 128 + max_db)
                            a_max = min(128 - max_da, c - 1)

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

    return [best_A, best_B]
