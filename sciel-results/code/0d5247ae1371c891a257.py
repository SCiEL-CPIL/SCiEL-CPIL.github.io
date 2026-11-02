"""Fresh design: tetracode direct product baseline + greedy covering reduction."""
import itertools
import time


def construct(q, n, radius):
    start = time.time()
    DEADLINE = 118.0  # wall-clock guard with safety margin under the 150s kill

    # --- Baseline: ternary tetracode (perfect radius-1 code, length 4) x full space ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base = [h + list(tail)
            for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]
    best = base

    # Fast integer encoding of length-n words over q
    pows = [1] * (n + 1)
    for i in range(1, n + 1):
        pows[i] = pows[i - 1] * q

    def enc(w):
        v = 0
        for i, x in enumerate(w):
            v += x * pows[i]
        return v

    # --- Precompute the covering ball (ints) for each baseline codeword ---
    balls = []
    for cw in base:
        s = {enc(cw)}
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                cw[pos] = val
                s.add(enc(cw))
            cw[pos] = orig
        balls.append(s)

    # --- Greedy reduction: drop a codeword iff its ball is covered by the rest ---
    # Priority by ball size (smallest first) so we shed redundant words cheaply.
    order = sorted(range(len(base)), key=lambda i: len(balls[i]))
    active = [True] * len(base)
    try:
        for idx in order:
            if time.time() > DEADLINE:
                break
            if not active[idx]:
                continue
            b = balls[idx]
            # Is every element of b covered by some other active codeword?
            covered = True
            for x in b:
                found = False
                for j in range(len(base)):
                    if j != idx and active[j] and x in balls[j]:
                        found = True
                        break
                if not found:
                    covered = False
                    break
            if covered:
                active[idx] = False
    except Exception:
        pass

    if time.time() > DEADLINE:
        return best

    reduced = [base[i] for i in range(len(base)) if active[i]]
    if len(reduced) < len(best):
        best = reduced
    return best
