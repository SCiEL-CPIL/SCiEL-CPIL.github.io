"""Rewritten program: local search to minimize |signed harmonic subsum| (nonzero)."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_N = 30
_DEN = [Fraction(1, i) for i in range(1, _N + 1)]

def _eval(c):
    """Return exact Fraction sum of c[i-1]/i."""
    s = Fraction(0)
    for i in range(_N):
        if c[i] != 0:
            s += c[i] * _DEN[i]
    return s

def construct():
    deadline = time.time() + 80.0
    best_c = None
    best_abs = None

    # Seed starting point
    seed = [0] * 28 + [1, -1]

    rng = random.Random(12345)

    def try_local_search(start, max_steps=4000):
        nonlocal best_c, best_abs
        c = list(start)
        s = _eval(c)
        # track current abs
        cur_abs = abs(s) if s != 0 else None
        # local best
        lb_c = list(c)
        lb_abs = cur_abs
        if lb_abs is not None and (best_abs is None or lb_abs < best_abs):
            best_abs = lb_abs
            best_c = list(c)
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return
            improved = False
            # find steepest improving flip
            best_gain = None
            best_i = -1
            best_nc = None
            for i in range(_N):
                # try each of the 2 other values
                for nv in (-1, 0, 1):
                    if nv == c[i]:
                        continue
                    # delta = (nv - c[i]) / (i+1)
                    delta = Fraction(nv - c[i], i + 1)
                    ns = s + delta
                    na = abs(ns) if ns != 0 else None
                    # we prefer smaller abs; if ns==0 skip (invalid)
                    if na is None:
                        continue
                    # gain: prefer smaller na; also allow moves that keep nonzero
                    if best_gain is None or na < best_gain:
                        best_gain = na
                        best_i = i
                        best_nc = nv
                        best_ns = ns
            if best_i >= 0 and best_gain < cur_abs:
                # apply
                c[best_i] = best_nc
                s = best_ns
                cur_abs = best_gain
                improved = True
                lb_c = list(c)
                lb_abs = cur_abs
                if best_abs is None or cur_abs < best_abs:
                    best_abs = cur_abs
                    best_c = list(c)
            else:
                # stuck: random kick to escape
                for _ in range(3):
                    if time.time() > deadline - 2.0:
                        return
                    ki = rng.randrange(_N)
                    c[ki] = rng.choice([-1, 0, 1])
                s = _eval(c)
                cur_abs = abs(s) if s != 0 else None
                if cur_abs is None:
                    # zero: nudge
                    ki = rng.randrange(_N)
                    c[ki] = rng.choice([-1, 1])
                    s = _eval(c)
                    cur_abs = abs(s)
                lb_c = list(c)
                lb_abs = cur_abs
                if best_abs is None or cur_abs < best_abs:
                    best_abs = cur_abs
                    best_c = list(c)

    # Start from seed
    try_local_search(seed)

    # Random restarts
    restarts = 0
    while time.time() < deadline - 3.0:
        # random start
        start = [rng.choice([-1, 0, 1]) for _ in range(_N)]
        # ensure nonzero sum
        s = _eval(start)
        if s == 0:
            start[0] = 1
        try_local_search(start, max_steps=2000)
        restarts += 1
        if restarts > 200000:
            break

    if best_c is None:
        return seed
    return best_c
# EVOLVE-BLOCK-END
