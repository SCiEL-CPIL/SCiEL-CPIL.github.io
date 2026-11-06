"""Doubly-covered prefix via local search over residue classes."""
import time
import numpy as np

_DEADLINE = time.time() + 135.0  # safety margin before 150s wall-clock kill


def _primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0]


def _coverage(counts, primes, max_l):
    # counts: array of length max_l+1 of current coverage
    # returns largest L with all 1..L covered >= 2
    # find first position with coverage < 2
    below = np.nonzero(counts[1:max_l + 1] < 2)[0]
    if below.size == 0:
        return max_l
    return int(below[0])


def _score(counts, primes, max_l):
    L = _coverage(counts, primes, max_l)
    if L >= max_l:
        return max_l
    # extend: check how far beyond L the >=2 condition holds
    tail = counts[L + 1:max_l + 1]
    if tail.size == 0:
        return L
    # find first <2 in tail
    bad = np.nonzero(tail < 2)[0]
    if bad.size == 0:
        return max_l
    return L + int(bad[0])


def _remove_add(counts, primes, idx, old_r, new_r, max_l):
    p = primes[idx]
    if old_r == new_r:
        return 0
    counts[old_r::p] -= 1
    counts[new_r::p] += 1
    return 0


def _init_counts(primes, residues, max_l):
    counts = np.zeros(max_l + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r::p] += 1
    return counts


def construct():
    primes = _primes_upto(1000)
    n = len(primes)  # 168
    max_l = 5000  # reduced from 20000 to keep O(N) ops fast

    # initial residues: cover ceil((i+2)/2) like seed, but we will improve
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    counts = _init_counts(primes, residues, max_l)
    best_score = _score(counts, primes, max_l)
    best_res = residues.copy()

    # local search: single-residue moves, accept if score improves
    order = np.arange(n)
    np.random.seed(0)

    improved = True
    stagnation = 0
    while improved and time.time() < _DEADLINE:
        improved = False
        np.random.shuffle(order)
        for idx in order:
            if time.time() >= _DEADLINE:
                break
            p = primes[idx]
            cur = residues[idx]
            cur_score = _score(counts, primes, max_l)
            # try candidate residues: focus on ones that help the current gap region
            # sample a few candidates plus neighbors
            cands = []
            # candidate that hits the first gap
            gap = cur_score
            if gap < max_l:
                # want x=gap+1 covered: residue (gap+1) % p
                cands.append((gap + 1) % p)
            # also try residue to cover gap+2
            if gap + 1 < max_l:
                cands.append((gap + 2) % p)
            # random candidates
            for _ in range(3):
                cands.append(np.random.randint(0, p))
            # current neighbor residues
            cands.append((cur + 1) % p)
            cands.append((cur - 1) % p)

            best_cand_score = cur_score
            best_cand = cur
            for c in cands:
                c = int(c) % p
                if c == cur:
                    continue
                _remove_add(counts, primes, idx, cur, c, max_l)
                s = _score(counts, primes, max_l)
                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = c
                _remove_add(counts, primes, idx, c, cur, max_l)
            if best_cand != cur:
                _remove_add(counts, primes, idx, cur, best_cand, max_l)
                residues[idx] = best_cand
                improved = True
                stagnation = 0
                if best_cand_score > best_score:
                    best_score = best_cand_score
                    best_res = residues.copy()
            else:
                stagnation += 1
        # if no improvement in a full pass, do a randomized restart-ish perturbation
        if not improved:
            if stagnation > 5 and time.time() < _DEADLINE - 20:
                # perturb a few residues to escape local optimum
                for _ in range(5):
                    idx = int(np.random.randint(0, n))
                    p = primes[idx]
                    old = residues[idx]
                    new = int(np.random.randint(0, p))
                    _remove_add(counts, primes, idx, old, new, max_l)
                    residues[idx] = new
                s = _score(counts, primes, max_l)
                if s < best_score - 50:
                    # revert perturbation if it hurt a lot
                    pass
                stagnation = 0

    # final polish: one more deterministic pass over all indices with all-residue scan
    # for the top few indices only if time allows
    if time.time() < _DEADLINE - 5:
        for idx in range(n):
            if time.time() >= _DEADLINE - 2:
                break
            p = primes[idx]
            cur = residues[idx]
            cur_score = _score(counts, primes, max_l)
            # full scan for small primes
            if p <= 30:
                for c in range(p):
                    if c == cur:
                        continue
                    _remove_add(counts, primes, idx, cur, c, max_l)
                    s = _score(counts, primes, max_l)
                    if s > cur_score:
                        cur_score = s
                        best_c = c
                    else:
                        best_c = cur
                    _remove_add(counts, primes, idx, c, cur, max_l)
                if best_c != cur:
                    _remove_add(counts, primes, idx, cur, best_c, max_l)
                    residues[idx] = best_c
                    if cur_score > best_score:
                        best_score = cur_score
                        best_res = residues.copy()

    # return the best found
    return [int(r) for r in best_res]
