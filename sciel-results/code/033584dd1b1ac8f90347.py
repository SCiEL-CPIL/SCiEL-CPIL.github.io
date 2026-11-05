import time
import math
import numpy as np

_T0 = time.time()
_DEADLINE = 125.0  # safety margin under the 125-150s kill

# --- primes <= 1000 (168 of them), increasing ---
def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]

PRIMES = _primes_upto(1000)
P = len(PRIMES)
PM = np.array(PRIMES, dtype=np.int64)
INV = np.array([np.int64(x) for x in np.modinv(PM, PM)], dtype=np.int64)


def _score(res):
    """Largest L with every integer 1..L covered by >=2 congruences (capped 1e5)."""
    L = 100000
    counts = np.zeros(L + 1, dtype=np.int32)
    for p, r in zip(PRIMES, res):
        counts[r::p] += 1
    ok = counts[1:] >= 2
    if ok.all():
        return L
    return int(np.argmax(~ok))


def construct():
    # ---------- Phase 1: forward greedy (feasible 2-cover) ----------
    res = [0] * P
    counts = np.zeros(100001, dtype=np.int32)
    for i, p in enumerate(PRIMES):
        best = 0
        bestcov = -1
        for r in range(p):
            seg = counts[r::p]
            c = int(np.sum(seg < 2))
            if c > bestcov:
                bestcov = c
                best = r
        res[i] = best
        counts[best::p] += 1

    best = res[:]
    bestL = _score(best)

    # ---------- Phase 2: backward coordinate descent ----------
    # coverage[x] = number of primes p with x == res_p (mod p), for x in 1..L
    cov = np.zeros(100001, dtype=np.int32)
    for p, r in zip(PRIMES, res):
        cov[r::p] += 1

    # deficit[x] = max(0, 2 - cov[x])  (how much more coverage x needs)
    deficit = np.maximum(0, 2 - cov)
    # running prefix sums of deficit, index 0..100000
    prefix = np.zeros(100001, dtype=np.int64)
    prefix[1:] = np.cumsum(deficit, dtype=np.int64)

    # order: largest primes first (they have the most residue choices)
    order = sorted(range(P), key=lambda i: -PRIMES[i])

    for it in range(200):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in order:
            p = PRIMES[i]
            r_old = res[i]
            # remove this prime's coverage
            cov[r_old::p] -= 1
            deficit = np.maximum(0, 2 - cov)
            prefix = np.zeros(100001, dtype=np.int64)
            prefix[1:] = np.cumsum(deficit, dtype=np.int64)

            # evaluate each candidate residue
            best_r = r_old
            best_gain = 0
            for r in range(p):
                # deficit on [r, r+p-1] = prefix[r+p-1] - prefix[r-1]
                hi = r + p - 1
                if hi > 100000:
                    hi = 100000
                gain = int(prefix[hi] - (prefix[r - 1] if r >= 1 else 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r

            if best_r != r_old:
                # apply best
                cov[best_r::p] += 1
                res[i] = best_r
                improved = True
            else:
                # restore old
                cov[r_old::p] += 1

        # after a full pass, recompute L and update best
        L = _score(res)
        if L > bestL:
            bestL = L
            best = res[:]
        if not improved:
            break

    return best


if __name__ == "__main__":
    r = construct()
    print(len(r), "residues, score =", _score(r))
