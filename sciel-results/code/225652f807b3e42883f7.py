"""Bit-parallel randomized local search with iterative compression for K_3(7,1)."""
import itertools
import random
import time
import numpy as np


def construct(q, n, radius):
    # This task is fixed at q=3, n=7, radius=1, but keep the signature general.
    t0 = time.time()
    DEADLINE = t0 + 138.0          # safety margin under the 150s wall-clock kill
    rng = random.Random(20240517)

    N = q ** n
    if N <= 0:
        return []
    if radius < 0:
        radius = 0

    # Precompute the radius-1 ball for every word as a bitset (Python int).
    # Only feasible for the small spaces this task uses.
    ball = [0] * N
    for idx in range(N):
        w = [0] * n
        v = idx
        for pos in range(n - 1, -1, -1):
            w[pos] = v % q
            v //= q
        b = 1 << idx
        for pos in range(n):
            for d in range(1, q):
                v2 = idx + (d - w[pos]) * (q ** pos)
                b |= 1 << v2
        ball[idx] = b

    # Start from the direct-sum construction (tetracode x full tail).
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    start = [h + list(tail) for h in tetracode
             for tail in itertools.product(range(3), repeat=n - 4)]
    code = list(start)
    cov = 0
    for c in code:
        cov |= ball[c]

    best = len(code)
    best_code = list(code)

    def try_delete(i):
        nonlocal code, cov, best, best_code
        b = ball[code[i]]
        if (cov & ~b) == (cov ^ b):   # ball fully covered by the rest
            code.pop(i)
            cov ^= b
            if len(code) < best:
                best = len(code)
                best_code = list(code)

    def try_merge(i, j):
        nonlocal code, cov, best, best_code
        ci = code[i]
        cj = code[j]
        bi = ball[ci]
        bj = ball[cj]
        others = cov ^ bi ^ bj
        union = bi | bj
        # Candidate centers: prefer the intersection of the two balls, else the smaller ball.
        inter = bi & bj
        if inter:
            cand = inter
        else:
            cand = bi if bi.bit_count() <= bj.bit_count() else bj
        bits = np.frombuffer(cand.to_bytes((cand.bit_length() + 7) // 8, "little"), dtype=np.uint64)
        # For each candidate center, the new coverage is others | ball[center];
        # the merge is valid iff the union is a subset of that new coverage.
        ok = np.zeros(len(bits), dtype=bool)
        for k in range(len(bits)):
            for b in range(8):
                bit = bits[k] >> b & 1
                if bit:
                    idx = k * 64 + b
                    if idx < N:
                        ok[k] |= (union & ~ball[idx]) == 0
        if ok.any():
            # Pick the candidate center that covers the most of the union (max overlap).
            ks = np.nonzero(ok)[0]
            best_k = -1
            best_val = -1
            for k in ks:
                for b in range(8):
                    bit = bits[k] >> b & 1
                    if bit:
                        idx = k * 64 + b
                        if idx < N:
                            val = (union & ball[idx]).bit_count()
                            if val > best_val:
                                best_val = val
                                best_k = k
            # Recover the concrete center index (lowest set bit among ok candidates).
            cands = np.nonzero(ok)[0]
            k0 = int(cands[0])
            center = None
            for k in cands:
                for b in range(8):
                    if (bits[k] >> b) & 1:
                        idx = k * 64 + b
                        if idx < N:
                            center = idx
                            break
                if center is not None:
                    break
            if center is None:
                return
            # Replace the later index first so indices stay valid.
            hi, lo = (i, j) if i > j else (j, i)
            code[hi] = center
            code.pop(lo)
            cov = (cov ^ bi ^ bj) | ball[center]
            if len(code) < best:
                best = len(code)
                best_code = list(code)

    # Phase A: greedy single-deletion sweep (cheap, removes obvious redundancy).
    for _ in range(200):
        if time.time() >= DEADLINE:
            break
        progressed = False
        for i in range(len(code) - 1, -1, -1):
            try_delete(i)
            progressed = True
        if not progressed:
            break

    # Phase B: randomized local search with deletion + pair-merge compression.
    while time.time() < DEADLINE:
        m = len(code)
        if m < best:
            best = m
            best_code = list(code)
        if m <= 2:
            break
        # Random pair-merge attempts.
        for _ in range(4000):
            if time.time() >= DEADLINE:
                break
            i = rng.randrange(m)
            j = rng.randrange(m)
            if i == j:
                continue
            try_merge(i, j)
            m = len(code)
        # Random deletion attempts.
        for _ in range(1500):
            if time.time() >= DEADLINE:
                break
            i = rng.randrange(len(code))
            try_delete(i)
        # Occasional full deletion sweep to clear accumulated redundancy.
        if time.time() < DEADLINE and len(code) % 50 == 0:
            for i in range(len(code) - 1, -1, -1):
                try_delete(i)

    return best_code


if __name__ == "__main__":
    res = construct(3, 7, 1)
    print("size:", len(res))
