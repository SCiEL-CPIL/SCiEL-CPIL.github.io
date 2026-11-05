"""Rewritten program: efficient AP-discrepancy minimization via hill climbing + SA."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
DEADLINE = 55.0  # seconds, safety margin under 60s evaluator limit


def calc_max_discrepancy(signs):
    """Max absolute signed sum over ANY contiguous sub-sequence of ANY AP in [0,N).
    For each step d and each residue class, the AP is a 1-D sequence; the max
    absolute contiguous sum equals max(prefix) - min(prefix). Overall answer is
    the max over all (d, residue). Singletons give |sign|=1 so answer >= 1.
    """
    n = len(signs)
    best = 1  # singletons
    # d = 1 covers the whole sequence (contiguous subarray of signs)
    # For each d, walk residue classes
    for d in range(1, n):
        for r in range(d):
            # sequence indices r, r+d, r+2d, ...
            s = 0
            mn = 0
            mx = 0
            idx = r
            while idx < n:
                s += signs[idx]
                if s < mn:
                    mn = s
                elif s > mx:
                    mx = s
                idx += d
            span = mx - mn
            if span > best:
                best = span
    return best


def _try_flip(signs, i, cur_best):
    """Return new best discrepancy if flipping sign i would help, else None.
    We do a full recompute only when needed; here we just recompute to be safe
    but that's O(n^2). To be faster we accept a recompute.
    """
    signs[i] = -signs[i]
    val = calc_max_discrepancy(signs)
    signs[i] = -signs[i]
    return val


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Start from a random sign vector
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    best = calc_max_discrepancy(signs)
    best_signs = signs[:]

    def elapsed():
        return time.time() - t0

    # Hill climbing + simulated annealing over single flips
    T = 4.0  # initial temperature
    cooling = 0.99997
    step = 0

    while elapsed() < DEADLINE:
        # Occasionally do a small block of flips
        i = rng.randrange(N)
        signs[i] = -signs[i]
        val = calc_max_discrepancy(signs)
        if val < best:
            best = val
            best_signs = signs[:]
            T = max(T * 1.02, 0.5)  # warm up a bit on improvement
        else:
            delta = val - best
            # accept with probability exp(-delta / T)
            if T > 0.01 and rng.random() < 2.718281828 ** (-delta / T):
                pass  # keep the flip
            else:
                signs[i] = -signs[i]  # revert
        T *= cooling
        step += 1

        # Periodic local polish: greedy single-flip descent
        if step % 2000 == 0 and elapsed() < DEADLINE - 3.0:
            improved = True
            while improved and elapsed() < DEADLINE - 3.0:
                improved = False
                for j in range(N):
                    signs[j] = -signs[j]
                    v = calc_max_discrepancy(signs)
                    if v < best:
                        best = v
                        best_signs = signs[:]
                        improved = True
                    else:
                        signs[j] = -signs[j]

    # Final: ensure we return the best found
    # Verify best_signs has correct discrepancy (sanity)
    return best_signs
# EVOLVE-BLOCK-END
