"""Hierarchical construction: dense greedy core + incremental local search."""
import time
import random
import numpy as np

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_upto(1000)  # 168 primes
_K = len(_PRIMES)
_MAX = 100000
_KCORE = 32          # number of smallest primes used for the dense core
_MCORE = 180         # target prefix to over-cover with the core


def _build_core():
    """Greedy covering-design: pick each small prime's residue to reduce the
    number of x in [1, MCORE] with coverage < 2."""
    cnt = np.zeros(_MAX + 1, dtype=np.int32)
    res = [0] * _K
    for i in range(_KCORE):
        p = _PRIMES[i]
        best_r, best_score = 0, -1
        for r in range(p):
            # positions in [1, MCORE] congruent to r mod p
            start = r if r >= 1 else r + p
            score = 0
            for x in range(start, _MCORE + 1, p):
                if cnt[x] < 2:
                    score += 1
            if score > best_score:
                best_score = score
                best_r = r
        res[i] = best_r
        cnt[best_r::p] += 1
    return res, cnt


def _init_tail(res):
    """Seed-style initialization for the tail primes (free to be improved)."""
    for i in range(_KCORE, _K):
        p = _PRIMES[i]
        res[i] = ((i + 2) // 2) % p


def _compute_L(cnt, start=0):
    L = start
    while L >= 1 and cnt[L] < 2:
        L -= 1
    while L < _MAX and cnt[L + 1] >= 2:
        L += 1
    return L


def construct():
    deadline = time.time() + 140.0
    random.seed(12345)

    res, cnt = _build_core()
    _init_tail(res)

    L = _compute_L(cnt)
    best_L = L
    best_res = res[:]

    # quick snapshot of core-only achievement
    def try_step():
        nonlocal L, best_L, best_res
        # bias toward tail (boundary-extending primes), allow some core moves
        if random.random() < 0.75:
            i = _KCORE + random.randrange(_K - _KCORE)
        else:
            i = random.randrange(_KCORE)
        p = _PRIMES[i]
        old = res[i]
        new = random.randrange(p)
        if new == old:
            return
        # apply
        cnt[old::p] -= 1
        cnt[new::p] += 1
        res[i] = new
        # incremental L
        while L >= 1 and cnt[L] < 2:
            L -= 1
        while L < _MAX and cnt[L + 1] >= 2:
            L += 1
        if L > best_L:
            best_L = L
            best_res = res[:]
        else:
            # occasional escape move; otherwise revert
            if random.random() < 0.04:
                # keep (exploration)
                pass
            else:
                cnt[new::p] -= 1
                cnt[old::p] += 1
                res[i] = old
                # L unchanged by revert (net zero), but re-sync cheaply
                while L >= 1 and cnt[L] < 2:
                    L -= 1
                while L < _MAX and cnt[L + 1] >= 2:
                    L += 1

    step = 0
    stuck = 0
    while time.time() < deadline:
        before = best_L
        for _ in range(200):
            if time.time() >= deadline:
                break
            try_step()
            step += 1
        if best_L > before:
            stuck = 0
        else:
            stuck += 1
            if stuck >= 12:
                # plateau: re-randomize tail to escape, keep best
                for i in range(_KCORE, _K):
                    res[i] = random.randrange(_PRIMES[i])
                # rebuild cnt from scratch to be safe
                cnt = np.zeros(_MAX + 1, dtype=np.int32)
                for i in range(_K):
                    cnt[res[i]::_PRIMES[i]] += 1
                L = _compute_L(cnt)
                stuck = 0

    # restore best configuration (return values only; cnt not needed)
    return best_res
