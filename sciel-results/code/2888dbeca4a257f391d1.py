"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 80.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum is 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _compute_factor_sets(nums):
    """For each pair (i,j), compute the set of prime factors of nums[i]+nums[j].
    Return a list of 12 sets, where set[i] contains all prime factors of sums involving index i."""
    n = len(nums)
    pair_factors = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf = _prime_factors_fast(s)
            pair_factors[i][j] = pf
            pair_factors[j][i] = pf
    return pair_factors

def _total_factor_count(pair_factors):
    """Count distinct primes across all pair factor sets."""
    all_primes = set()
    n = len(pair_factors)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pair_factors[i][j]
    return len(all_primes)

def _build_prime_count(pair_factors):
    """Build a dict mapping each prime to the number of pairs (i<j) containing it."""
    n = len(pair_factors)
    from collections import defaultdict
    pc = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            for p in pair_factors[i][j]:
                pc[p] += 1
    return pc

def construct():
    random.seed(12345)
    n = 12

    # Initial candidates
    candidates = []
    candidates.append(list(range(2, 25, 2)))
    candidates.append(list(range(6, 78, 6)))
    candidates.append(list(range(12, 156, 12)))
    candidates.append([2**i for i in range(1, 13)])
    for _ in range(8):
        candidates.append(sorted(random.sample(range(1, 10001), 12)))
    # Additional structured candidates
    candidates.append(list(range(3, 39, 3)))
    candidates.append(list(range(4, 53, 4)))
    candidates.append(list(range(10, 131, 10)))

    best_candidate = None
    best_score = float('inf')
    best_pair_factors = None

    for cand in candidates:
        if _deadline_hit():
            break
        if len(set(cand)) != n:
            continue
        if not all(1 <= x <= 10000 for x in cand):
            continue
        pf = _compute_factor_sets(cand)
        score = _total_factor_count(pf)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
            best_pair_factors = pf

    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_pair_factors = _compute_factor_sets(best_candidate)
        best_score = _total_factor_count(best_pair_factors)

    current = best_candidate[:]
    current_pf = best_pair_factors
    current_score = best_score
    prime_count = _build_prime_count(current_pf)

    def _try_replace(idx, new_val):
        """Try replacing current[idx] with new_val. Returns True if accepted."""
        nonlocal current, current_pf, current_score, prime_count
        if new_val in current:
            return False
        # Compute new pairs involving idx
        old_primes_set = set()
        new_primes_set = set()
        for j in range(n):
            if j == idx:
                continue
            old_pf = current_pf[idx][j]
            new_pf = _prime_factors_fast(new_val + current[j])
            old_primes_set |= old_pf
            new_primes_set |= new_pf

        # Compute delta in distinct prime count
        # A prime p is removed from the global set if its count goes to 0
        # A prime p is added if its count goes from 0 to >0
        delta = 0
        for p in old_primes_set:
            if prime_count.get(p, 0) == 1:
                delta -= 1
        for p in new_primes_set:
            if prime_count.get(p, 0) == 0:
                delta += 1

        new_score = current_score + delta
        if new_score < current_score:
            # Accept
            for j in range(n):
                if j == idx:
                    continue
                old_pf = current_pf[idx][j]
                new_pf = _prime_factors_fast(new_val + current[j])
                for p in old_pf:
                    prime_count[p] -= 1
                    if prime_count[p] == 0:
                        del prime_count[p]
                for p in new_pf:
                    if p not in prime_count:
                        prime_count[p] = 0
                    prime_count[p] += 1
                current_pf[idx][j] = new_pf
                current_pf[j][idx] = new_pf
            current[idx] = new_val
            current_score = new_score
            return True
        return False

    def _try_swap(i, j):
        """Try swapping current[i] and current[j]. Returns True if accepted."""
        nonlocal current, current_pf, current_score, prime_count
        if i == j:
            return False
        vi, vj = current[i], current[j]
        if vi == vj:
            return False
        # After swap: i gets vj, j gets vi
        # Pairs affected: (i,k) for k!=i,j, (j,k) for k!=i,j, and (i,j)
        # Compute old and new primes for affected pairs
        affected_pairs = []
        for k in range(n):
            if k != i and k != j:
                affected_pairs.append((i, k))
                affected_pairs.append((j, k))
        affected_pairs.append((i, j))

        old_primes_all = set()
        new_primes_all = set()
        for (a, b) in affected_pairs:
            old_pf = current_pf[a][b]
            if a == i:
                new_val_a = vj
            elif a == j:
                new_val_a = vi
            else:
                new_val_a = current[a]
            if b == i:
                new_val_b = vj
            elif b == j:
                new_val_b = vi
            else:
                new_val_b = current[b]
            new_pf = _prime_factors_fast(new_val_a + new_val_b)
            old_primes_all |= old_pf
            new_primes_all |= new_pf

        delta = 0
        for p in old_primes_all:
            if prime_count.get(p, 0) == 1:
                delta -= 1
        for p in new_primes_all:
            if prime_count.get(p, 0) == 0:
                delta += 1

        new_score = current_score + delta
        if new_score < current_score:
            # Accept swap
            for (a, b) in affected_pairs:
                old_pf = current_pf[a][b]
                if a == i:
                    new_val_a = vj
                elif a == j:
                    new_val_a = vi
                else:
                    new_val_a = current[a]
                if b == i:
                    new_val_b = vj
                elif b == j:
                    new_val_b = vi
                else:
                    new_val_b = current[b]
                new_pf = _prime_factors_fast(new_val_a + new_val_b)
                for p in old_pf:
                    prime_count[p] -= 1
                    if prime_count[p] == 0:
                        del prime_count[p]
                for p in new_pf:
                    if p not in prime_count:
                        prime_count[p] = 0
                    prime_count[p] += 1
                current_pf[a][b] = new_pf
                current_pf[b][a] = new_pf
            current[i], current[j] = vj, vi
            current_score = new_score
            return True
        return False

    # Hill climbing with random restarts
    for restart in range(300):
        if _deadline_hit():
            break

        improved = True
        while improved and not _deadline_hit():
            improved = False

            # Try single-element replacements
            for idx in range(n):
                if _deadline_hit():
                    break
                old_val = current[idx]
                # Small perturbations
                tried = set()
                for d in range(-30, 31):
                    v = old_val + d
                    if 1 <= v <= 10000 and v != old_val and v not in current and v not in tried:
                        tried.add(v)
                        if _try_replace(idx, v):
                            improved = True
                            break
                if improved:
                    break
                # Random values
                for _ in range(15):
                    v = random.randint(1, 10000)
                    if v not in current and v not in tried:
                        tried.add(v)
                        if _try_replace(idx, v):
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # Try swaps
            for i in range(n):
                if _deadline_hit():
                    break
                for j in range(i + 1, n):
                    if _deadline_hit():
                        break
                    if _try_swap(i, j):
                        improved = True
                        break
                if improved:
                    break

        # If stuck, try a random perturbation
        if not _deadline_hit():
            new_cand = current[:]
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, n - 1)
                new_cand[idx] = random.randint(1, 10000)
            if len(set(new_cand)) == n and all(1 <= x <= 10000 for x in new_cand):
                new_pf = _compute_factor_sets(new_cand)
                new_score = _total_factor_count(new_pf)
                if new_score < current_score:
                    current = new_cand
                    current_pf = new_pf
                    current_score = new_score
                    prime_count = _build_prime_count(current_pf)

    # Final polish: tighter local search with smaller steps
    if not _deadline_hit():
        improved = True
        while improved and not _deadline_hit():
            improved = False
            for idx in range(n):
                if _deadline_hit():
                    break
                old_val = current[idx]
                for d in range(-10, 11):
                    v = old_val + d
                    if 1 <= v <= 10000 and v != old_val and v not in current:
                        if _try_replace(idx, v):
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            for i in range(n):
                if _deadline_hit():
                    break
                for j in range(i + 1, n):
                    if _deadline_hit():
                        break
                    if _try_swap(i, j):
                        improved = True
                        break
                if improved:
                    break

    # Final verification
    result = current
    assert len(set(result)) == n
    assert all(1 <= x <= 10000 for x in result)

    return result

# EVOLVE-BLOCK-END
