"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill

    n = 12
    max_val = 10000
    limit = 2 * max_val

    # Precompute smallest prime factor (SPF) sieve up to 2*max_val
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute prime-factor sets for every value 2..limit (frozenset for fast union)
    pf_sets = [None] * (limit + 1)
    pf_sets[0] = frozenset()
    pf_sets[1] = frozenset()
    for x in range(2, limit + 1):
        factors = set()
        y = x
        while y > 1:
            p = spf[y]
            factors.add(p)
            while y % p == 0:
                y //= p
        pf_sets[x] = frozenset(factors)

    # Precompute all pairwise index pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def score(nums):
        """Count distinct prime factors across all pairwise sums (O(1) per sum via lookup)."""
        primes = set()
        for i, j in pairs:
            primes |= pf_sets[nums[i] + nums[j]]
        return len(primes)

    def score_and_primes(nums):
        """Return (count, set) of distinct prime factors."""
        primes = set()
        for i, j in pairs:
            primes |= pf_sets[nums[i] + nums[j]]
        return len(primes), primes

    # Initial best: small numbers
    best = list(range(1, 13))
    best_score = score(best)

    # Strategy 1: Arithmetic progressions
    for a in range(1, 300):
        for d in range(1, 300):
            if time.time() > deadline:
                break
            nums = [a + i * d for i in range(n)]
            if nums[-1] > max_val or nums[0] <= 0:
                continue
            sc = score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
        if time.time() > deadline:
            break

    # Strategy 2: Multiples of a base
    for base in range(2, 800):
        if time.time() > deadline:
            break
        for sm in range(1, 12):
            nums = [base * (sm + i) for i in range(n)]
            if nums[-1] > max_val:
                break
            sc = score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]

    # Strategy 3: Smooth numbers (2^a * 3^b * 5^c * 7^d)
    smooth = set()
    for a in range(0, 14):
        v1 = 2 ** a
        if v1 > max_val:
            break
        for b in range(0, 10):
            v2 = v1 * (3 ** b)
            if v2 > max_val:
                break
            for c in range(0, 7):
                v3 = v2 * (5 ** c)
                if v3 > max_val:
                    break
                for d in range(0, 5):
                    v4 = v3 * (7 ** d)
                    if v4 > max_val:
                        break
                    smooth.add(v4)
    smooth = sorted(smooth)
    if len(smooth) >= n:
        nums = smooth[:n]
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]
        nums = smooth[-n:]
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]
        for start_idx in range(0, max(1, len(smooth) - n), 20):
            if time.time() > deadline:
                break
            nums = smooth[start_idx:start_idx + n]
            sc = score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]

    # Strategy 4: All even numbers
    for start_even in range(2, 200, 2):
        if time.time() > deadline:
            break
        nums = [start_even + 2 * i for i in range(n)]
        if nums[-1] > max_val:
            continue
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]

    # Strategy 5: All multiples of 4
    for start_m4 in range(4, 400, 4):
        if time.time() > deadline:
            break
        nums = [start_m4 + 4 * i for i in range(n)]
        if nums[-1] > max_val:
            continue
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]

    # Strategy 6: All multiples of 6
    for start_m6 in range(6, 600, 6):
        if time.time() > deadline:
            break
        nums = [start_m6 + 6 * i for i in range(n)]
        if nums[-1] > max_val:
            continue
        sc = score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]

    # Strategy 7: Structured local search — target prime elimination
    if time.time() < deadline - 3:
        random.seed(42)
        current = best[:]
        current_score, current_primes = score_and_primes(current)

        # For each element i, precompute which primes are contributed ONLY by sums
        # involving i (i.e., primes that would disappear if we could change element i)
        def compute_i_only_primes(nums, primes):
            """For each index i, return set of primes that appear ONLY in sums involving i."""
            i_only = [set() for _ in range(n)]
            # For each prime, find which indices it appears in
            prime_indices = {}
            for p in primes:
                idxs = set()
                for i, j in pairs:
                    if p in pf_sets[nums[i] + nums[j]]:
                        idxs.add(i)
                        idxs.add(j)
                prime_indices[p] = idxs
            # A prime is "i-only" if it appears in sums involving i but in no other
            # sums that don't involve i. More precisely: if removing all sums with i
            # eliminates p, then p is i-only.
            for i in range(n):
                # Primes in sums involving i
                with_i = set()
                for i, j in pairs:
                    with_i |= pf_sets[nums[i] + nums[j]]
                # Primes in sums NOT involving i
                without_i = set()
                for ii, jj in pairs:
                    if i not in (ii, jj):
                        without_i |= pf_sets[nums[ii] + nums[jj]]
                i_only[i] = with_i - without_i
            return i_only

        i_only = compute_i_only_primes(current, current_primes)
        no_improve = 0
        max_no_improve = 5000

        while time.time() < deadline - 2 and no_improve < max_no_improve:
            # Pick the index with the most "only" primes to try to eliminate
            best_i = max(range(n), key=lambda k: len(i_only[k]))
            if len(i_only[best_i]) == 0:
                # No single-element prime to target; try random
                best_i = random.randint(0, n - 1)

            old_val = current[best_i]
            target_primes = i_only[best_i]

            # Generate candidate values that avoid the target primes in all sums with best_i
            found = False
            for attempt in range(200):
                # Bias toward smooth/multiple values
                r = random.random()
                if r < 0.3:
                    new_val = random.randint(1, max_val // 6) * 6
                elif r < 0.5:
                    new_val = random.randint(1, max_val // 4) * 4
                elif r < 0.7:
                    new_val = random.randint(1, max_val // 2) * 2
                elif r < 0.85:
                    new_val = random.randint(1, max_val // 3) * 3
                else:
                    new_val = random.randint(1, max_val)

                if new_val > max_val or new_val <= 0:
                    continue
                if new_val in current:
                    continue

                # Quick check: do any new sums introduce target_primes?
                ok = True
                for j in range(n):
                    if j == best_i:
                        continue
                    s = new_val + current[j]
                    if target_primes & pf_sets[s]:
                        ok = False
                        break
                if not ok:
                    continue

                # Full score
                current[best_i] = new_val
                sc, new_primes = score_and_primes(current)
                if sc < current_score:
                    current_score = sc
                    current_primes = new_primes
                    no_improve = 0
                    found = True
                    if current_score < best_score:
                        best_score = current_score
                        best = current[:]
                    # Recompute i_only
                    i_only = compute_i_only_primes(current, current_primes)
                    break
                else:
                    current[best_i] = old_val
                    no_improve += 1

            if found:
                continue
            # If no improvement found in this round, try a different index
            no_improve += 1

    return best
# EVOLVE-BLOCK-END
