"""Rewritten program: integer LCM-scaled local search for strict underpacking."""
# EVOLVE-BLOCK-START
import math
import random
import time

_DENOMS = list(range(2, 49))  # denominators 2..48
_N = len(_DENOMS)
_L = 1
for _d in _DENOMS:
    _L = _L // math.gcd(_L, _d) * _d
# _L is lcm(2..48); each 1/d == _L//d exactly

def _sum_int(mask):
    s = 0
    for i in range(_N):
        if mask >> i & 1:
            s += _L // _DENOMS[i]
    return s

def _valid(mask):
    # nonempty, sum strictly below 1 -> 0 < sum < L
    return mask != 0 and 0 < _sum_int(mask) < _L

def _deficit(mask):
    return _L - _sum_int(mask)  # positive when valid

def _greedy_start():
    mask = 0
    total = 0
    for i, d in enumerate(_DENOMS):
        v = _L // d
        if total + v < _L:
            mask |= (1 << i)
            total += v
    return mask

def _local_search(mask, deadline):
    best_mask = mask
    best_def = _deficit(mask)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1-add
        for i in range(_N):
            if not (mask >> i & 1):
                nm = mask | (1 << i)
                s = _sum_int(nm)
                if s < _L:
                    d = _L - s
                    if d < best_def:
                        best_def = d
                        best_mask = nm
                        improved = True
        if time.time() >= deadline:
            break
        # 1-remove
        for i in range(_N):
            if mask >> i & 1:
                nm = mask & ~(1 << i)
                if nm == 0:
                    continue
                s = _sum_int(nm)
                if s < _L:
                    d = _L - s
                    if d < best_def:
                        best_def = d
                        best_mask = nm
                        improved = True
        if time.time() >= deadline:
            break
        # 1-swap
        for i in range(_N):
            if mask >> i & 1:
                for j in range(_N):
                    if not (mask >> j & 1):
                        nm = (mask & ~(1 << i)) | (1 << j)
                        s = _sum_int(nm)
                        if s < _L:
                            d = _L - s
                            if d < best_def:
                                best_def = d
                                best_mask = nm
                                improved = True
        if time.time() >= deadline:
            break
        # 2-for-1 swap: remove one, add two (net size +1)
        for i in range(_N):
            if not (mask >> i & 1):
                continue
            for j in range(_N):
                if mask >> j & 1:
                    continue
                for k in range(j + 1, _N):
                    if mask >> k & 1:
                        continue
                    nm = (mask & ~(1 << i)) | (1 << j) | (1 << k)
                    s = _sum_int(nm)
                    if s < _L:
                        d = _L - s
                        if d < best_def:
                            best_def = d
                            best_mask = nm
                            improved = True
        if time.time() >= deadline:
            break
        # 1-for-2 swap: remove two, add one (net size -1)
        for i in range(_N):
            if not (mask >> i & 1):
                continue
            for j in range(i + 1, _N):
                if not (mask >> j & 1):
                    continue
                for k in range(_N):
                    if mask >> k & 1:
                        continue
                    nm = (mask & ~(1 << i)) & ~(1 << j) | (1 << k)
                    s = _sum_int(nm)
                    if s < _L:
                        d = _L - s
                        if d < best_def:
                            best_def = d
                            best_mask = nm
                            improved = True
        mask = best_mask
    return best_mask, best_def

def construct():
    start = time.time()
    # safety margin: stop search at 80s, return by ~85s
    deadline = start + 80.0
    best_mask = _greedy_start()
    best_def = _deficit(best_mask)

    # First, local search from greedy
    best_mask, best_def = _local_search(best_mask, deadline)

    # Random restarts
    rng = random.Random(12345)
    while time.time() < deadline:
        # build a random feasible starting set
        mask = 0
        total = 0
        order = list(range(_N))
        rng.shuffle(order)
        for i in order:
            v = _L // _DENOMS[i]
            if total + v < _L:
                mask |= (1 << i)
                total += v
        if mask == 0:
            continue
        # local search
        lm, ld = _local_search(mask, deadline)
        if ld < best_def:
            best_def = ld
            best_mask = lm
        # hard stop margin
        if time.time() > deadline - 5.0:
            break

    # Convert mask to zero-based indices
    indices = [i for i in range(_N) if best_mask >> i & 1]
    return indices
# EVOLVE-BLOCK-END
