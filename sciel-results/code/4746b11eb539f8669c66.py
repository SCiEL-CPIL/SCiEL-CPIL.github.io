"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill

    # Precompute prime factorization (set of prime factors) for all odd numbers in [3, 255]
    def prime_factors(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        if n % 2 == 0:
            factors.add(2)
            while n % 2 == 0:
                n //= 2
        p = 3
        while p * p <= n:
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
            p += 2
        if n > 1:
            factors.add(n)
        return factors

    # Precompute for all odd sums in [3, 255]
    pf_cache = {}
    for n in range(3, 256, 2):
        pf_cache[n] = prime_factors(n)

    def coprime_check_fast(sums_set):
        """Fast pairwise coprime check using precomputed prime factors."""
        slist = list(sums_set)
        seen_primes = set()
        for s in slist:
            if s not in pf_cache:
                # Shouldn't happen for odd sums in range
                pf = prime_factors(s)
            else:
                pf = pf_cache[s]
            if pf & seen_primes:
                return False
            seen_primes |= pf
        return True

    odd_vals = [v for v in range(1, 129) if v % 2 == 1]
    even_vals = [v for v in range(1, 129) if v % 2 == 0]

    best_score = 0
    best_A = None
    best_B = None

    def try_A_B(A, B):
        """Check if A,B gives 9 distinct pairwise-coprime sums. Returns count."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) != 9:
            return 0
        if coprime_check_fast(sums):
            return 9
        return 0

    # Phase 1: Backtracking search over A, then greedily pick B
    # Sort odd values by "diversity" - prefer values that produce sums with few small prime factors
    def diversity_score(val):
        """Higher is better: sums tend to have fewer small prime factors."""
        score = 0
        for b in even_vals[:20]:
            s = val + b
            if s <= 255 and s % 2 == 1:
                pf = pf_cache.get(s, prime_factors(s))
                # Penalize if small primes (3,5,7) are present
                penalty = 0
                for p in (3, 5, 7, 11, 13):
                    if p in pf:
                        penalty += 1
                score += (5 - penalty)
        return score

    sorted_odds = sorted(odd_vals, key=diversity_score, reverse=True)
    sorted_evens = sorted(even_vals, key=diversity_score, reverse=True)

    # Phase 1a: Systematic search with incremental B construction
    top_odds = sorted_odds[:25]
    top_evens = sorted_evens[:30]

    for a1 in top_odds[:12]:
        if time.time() > deadline - 5:
            break
        for a2 in top_odds[:12]:
            if a2 <= a1:
                continue
            if time.time() > deadline - 5:
                break
            for a3 in top_odds[:12]:
                if a3 <= a2:
                    continue
                if time.time() > deadline - 5:
                    break
                A = [a1, a2, a3]

                # For each candidate b, compute the 3 sums and their prime factors
                # Then try to pick 3 b's such that all 9 sums are pairwise coprime
                b_info = []
                for b in top_evens:
                    sums_for_b = [a + b for a in A]
                    # All sums must be in valid range
                    if any(s < 3 or s > 255 for s in sums_for_b):
                        continue
                    # Get prime factor sets
                    pfs = [pf_cache.get(s, prime_factors(s)) for s in sums_for_b]
                    # Check internal coprimality of the 3 sums
                    if pfs[0] & pfs[1] or pfs[0] & pfs[2] or pfs[1] & pfs[2]:
                        continue
                    b_info.append((b, sums_for_b, pfs))

                if len(b_info) < 3:
                    continue

                # Try to pick 3 from b_info such that all 9 sums are pairwise coprime
                # Use backtracking: pick b1, then b2 must not share primes with b1's sums, etc.
                found = False
                for i in range(len(b_info)):
                    if time.time() > deadline - 5:
                        break
                    b1, s1, pf1 = b_info[i]
                    seen1 = set()
                    for pf in pf1:
                        seen1 |= pf

                    for j in range(i + 1, len(b_info)):
                        b2, s2, pf2 = b_info[j]
                        # Check b2's sums don't share primes with b1's sums
                        ok = True
                        for pf in pf2:
                            if pf & seen1:
                                ok = False
                                break
                        if not ok:
                            continue
                        seen12 = seen1 | set()
                        for pf in pf2:
                            seen12 |= pf

                        for k in range(j + 1, len(b_info)):
                            b3, s3, pf3 = b_info[k]
                            ok3 = True
                            for pf in pf3:
                                if pf & seen12:
                                    ok3 = False
                                    break
                            if not ok3:
                                continue

                            B = [b1, b2, b3]
                            sums = set()
                            for a in A:
                                for b in B:
                                    sums.add(a + b)
                            if len(sums) == 9 and coprime_check_fast(sums):
                                best_score = 9
                                best_A = A[:]
                                best_B = B[:]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if best_score >= 9:
                break
        if best_score >= 9:
            break

    if best_score < 9:
        # Phase 1b: Wider systematic search
        for a1 in top_odds[:18]:
            if time.time() > deadline - 5:
                break
            for a2 in top_odds[:18]:
                if a2 <= a1:
                    continue
                if time.time() > deadline - 5:
                    break
                for a3 in top_odds[:18]:
                    if a3 <= a2:
                        continue
                    if time.time() > deadline - 5:
                        break
                    A = [a1, a2, a3]

                    b_info = []
                    for b in top_evens[:40]:
                        sums_for_b = [a + b for a in A]
                        if any(s < 3 or s > 255 for s in sums_for_b):
                            continue
                        pfs = [pf_cache.get(s, prime_factors(s)) for s in sums_for_b]
                        if pfs[0] & pfs[1] or pfs[0] & pfs[2] or pfs[1] & pfs[2]:
                            continue
                        b_info.append((b, sums_for_b, pfs))

                    if len(b_info) < 3:
                        continue

                    found = False
                    for i in range(min(len(b_info), 25)):
                        if time.time() > deadline - 5:
                            break
                        b1, s1, pf1 = b_info[i]
                        seen1 = set()
                        for pf in pf1:
                            seen1 |= pf

                        for j in range(i + 1, min(len(b_info), 35)):
                            b2, s2, pf2 = b_info[j]
                            ok = True
                            for pf in pf2:
                                if pf & seen1:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            seen12 = set(seen1)
                            for pf in pf2:
                                seen12 |= pf

                            for k in range(j + 1, min(len(b_info), 45)):
                                b3, s3, pf3 = b_info[k]
                                ok3 = True
                                for pf in pf3:
                                    if pf & seen12:
                                        ok3 = False
                                        break
                                if not ok3:
                                    continue

                                B = [b1, b2, b3]
                                sums = set()
                                for a in A:
                                    for b in B:
                                        sums.add(a + b)
                                if len(sums) == 9 and coprime_check_fast(sums):
                                    best_score = 9
                                    best_A = A[:]
                                    best_B = B[:]
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if best_score >= 9:
                    break
            if best_score >= 9:
                break

    if best_score < 9:
        # Phase 2: Random search
        import random
        random.seed(42)
        for _ in range(200000):
            if time.time() > deadline - 3:
                break
            A = random.sample(odd_vals, 3)
            B = random.sample(even_vals, 3)
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9 and coprime_check_fast(sums):
                best_score = 9
                best_A = A[:]
                best_B = B[:]
                break

    if best_score < 9:
        # Phase 3: Local search / hill climbing
        import random
        random.seed(7)
        # Start from a reasonable seed
        A = [3, 9, 27]
        B = [2, 6, 18]
        # Ensure distinct
        while len(set(A)) < 3:
            A[random.randint(0,2)] = random.choice(odd_vals)
        while len(set(B)) < 3:
            B[random.randint(0,2)] = random.choice(even_vals)

        for _ in range(300000):
            if time.time() > deadline - 2:
                break
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                A[idx] = random.choice(odd_vals)
            else:
                idx = random.randint(0, 2)
                B[idx] = random.choice(even_vals)

            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            if any(a < 1 or a > 128 for a in A) or any(b < 1 or b > 128 for b in B):
                continue

            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9 and coprime_check_fast(sums):
                best_score = 9
                best_A = A[:]
                best_B = B[:]
                break

    if best_A is not None and best_B is not None:
        return [best_A, best_B]
    else:
        # Fallback
        return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
