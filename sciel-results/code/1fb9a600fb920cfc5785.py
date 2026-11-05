"""Greedy + local search for prime residue covering with reciprocal budget 1."""
import time
import math
from fractions import Fraction
import numpy as np

_DEADLINE = 140.0  # safety margin under 150s wall-clock

def _time_ok(t0):
    return time.time() - t0 < _DEADLINE

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _coverage_len(covered):
    # covered is a bool numpy array of length N; return largest L s.t. 1..L all covered
    # We want the first gap in 1..N
    # Use argmax on ~covered shifted
    # covered[0] corresponds to integer 1
    if covered.all():
        return len(covered)
    gaps = np.flatnonzero(~covered)
    return int(gaps[0])  # first uncovered index i means integers 1..i are covered, so L=i

def _add(covered, p, r):
    # integer n is covered if n ≡ r mod p, i.e. n = r + k*p
    # In 0-indexed (integer n+1), covered index i (0-based) => integer i+1
    # integer i+1 ≡ r mod p => i+1 ≡ r mod p => i ≡ r-1 mod p
    start = (r - 1) % p
    covered[start::p] = True

def _remove(covered, p, r, all_classes):
    # Recompute: remove this class, then re-add others
    # Simpler: rebuild from scratch is too slow; instead recompute coverage for the slice
    # We'll just recompute the whole coverage for candidate evaluation in local search via a helper
    pass

def _build_coverage(classes, N):
    covered = np.zeros(N, dtype=bool)
    for p, r in classes:
        start = (r - 1) % p
        covered[start::p] = True
    return covered

def _coverage_len_from(covered):
    if covered.all():
        return len(covered)
    gaps = np.flatnonzero(~covered)
    return int(gaps[0])

def _eval_greedy(primes, N, t0):
    # Start with empty, add greedily by max new coverage, tie-break by small 1/p
    classes = []
    covered = np.zeros(N, dtype=bool)
    budget = Fraction(0, 1)
    used = set()
    # Precompute reciprocals as Fractions
    recips = {p: Fraction(1, p) for p in primes}
    best_len = 0
    while _time_ok(t0):
        best_gain = 0
        best_p = None
        best_r = None
        best_recip = None
        for p in primes:
            if p in used:
                continue
            # Try each residue, find max new coverage
            # We can vectorize: for each r, new = count of ~covered[r-1 mod p :: p]
            # Do it in a loop over r but vectorized count
            # To save time, skip if 1/p would exceed budget
            if budget + recips[p] > 1:
                continue
            # Compute gains for all r at once
            # gain[r] = number of i in [0,N) with i ≡ r-1 mod p and not covered
            # Let's compute using a temporary
            notcov = ~covered
            # For each r, indices are (r-1)%p, (r-1)%p + p, ...
            # We can compute total available per residue
            # Use: for residue class start s, count notcov[s::p]
            # We'll loop r but it's fast enough for moderate primes
            # Optimize: compute all at once using reshape trick is complex; loop r
            best_r_local = -1
            best_gain_local = -1
            for r in range(p):
                start = (r - 1) % p
                g = int(notcov[start::p].sum())
                if g > best_gain_local:
                    best_gain_local = g
                    best_r_local = r
            if best_gain_local > best_gain:
                best_gain = best_gain_local
                best_p = p
                best_r = best_r_local
                best_recip = recips[p]
        if best_p is None or best_gain <= 0:
            break
        _add(covered, best_p, best_r)
        classes.append([best_p, best_r])
        used.add(best_p)
        budget += best_recip
        best_len = _coverage_len_from(covered)
        if best_len >= N:
            break
    return classes, best_len

def _try_local_search(classes, primes, N, t0):
    # Try to extend coverage by swapping classes
    covered = _build_coverage(classes, N)
    cur_len = _coverage_len_from(covered)
    budget = sum(Fraction(1, p) for p, r in classes)
    used = set(p for p, r in classes)
    recips = {p: Fraction(1, p) for p in primes}
    improved = True
    while improved and _time_ok(t0):
        improved = False
        # For each class in use, try removing it and adding a different one
        for idx, (p_old, r_old) in enumerate(classes):
            if not _time_ok(t0):
                break
            # Remove old
            # Rebuild coverage without this class
            classes_without = classes[:idx] + classes[idx+1:]
            covered_wo = _build_coverage(classes_without, N)
            budget_wo = budget - recips[p_old]
            # Try adding each unused prime
            for p_new in primes:
                if p_new in used and p_new != p_old:
                    continue
                if p_new == p_old:
                    continue
                if budget_wo + recips[p_new] > 1:
                    continue
                # Find best residue for p_new on top of covered_wo
                notcov = ~covered_wo
                best_r_new = -1
                best_gain = -1
                for r in range(p_new):
                    start = (r - 1) % p_new
                    g = int(notcov[start::p_new].sum())
                    if g > best_gain:
                        best_gain = g
                        best_r_new = r
                if best_gain > 0:
                    # Evaluate new length
                    # Build new coverage
                    new_classes = classes_without + [[p_new, best_r_new]]
                    new_covered = covered_wo.copy()
                    start = (best_r_new - 1) % p_new
                    new_covered[start::p_new] = True
                    new_len = _coverage_len_from(new_covered)
                    if new_len > cur_len:
                        classes = new_classes
                        covered = new_covered
                        cur_len = new_len
                        budget = budget_wo + recips[p_new]
                        used = set(p for p, r in classes)
                        improved = True
                        break
            if improved:
                break
    return classes, cur_len

def construct():
    t0 = time.time()
    N = 100000  # cap
    primes = _primes_upto(1000)
    # Greedy
    classes, best_len = _eval_greedy(primes, N, t0)
    # Local search
    if _time_ok(t0):
        classes, ls_len = _try_local_search(classes, primes, N, t0)
        if ls_len > best_len:
            best_len = ls_len
    # Final coverage length
    covered = _build_coverage(classes, N)
    final_len = _coverage_len_from(covered)
    if final_len > best_len:
        best_len = final_len
    # Verify budget
    total = sum(Fraction(1, p) for p, r in classes)
    assert total <= 1, f"Budget exceeded: {total}"
    # Verify distinct primes and valid residues
    pset = set()
    for p, r in classes:
        assert 2 <= p <= 1000
        assert 0 <= r < p
        assert p not in pset
        pset.add(p)
    return classes
