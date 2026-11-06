import time
import random

_N = 256
_DEADLINE = 80.0  # safety margin before the 90s kill

# Precompute per-element masks (as Python ints / bitsets over positions 0..255).
# 3ap_mask[x]: set of y such that (x, y) forms a 3-AP with x the largest element.
# 4ap_mask[x]: set of y such that (x, y) completes a 4-AP (x largest of the 4).
_3ap_mask = [0] * (_N + 1)
_4ap_mask = [0] * (_N + 1)
for x in range(1, _N + 1):
    i = x - 1
    m3 = 0
    m4 = 0
    for d in range(1, x):
        a = x - d
        if a < 1:
            break
        # 3-AP: (a, x-d, x)
        m3 |= 1 << (a - 1)
        # 4-AP: (a, x-2d, x-3d, x)
        if x - 3 * d >= 1:
            m4 |= 1 << (a - 1)
    _3ap_mask[x] = m3
    _4ap_mask[x] = m4


def _popcount(v):
    return bin(v).count("1")


def _count_3aps(S):
    # S is a bitmask; count 3-APs = sum over x in S of popcount(S & 3ap_mask[x])
    total = 0
    x = 1
    while x <= _N:
        if (S >> (x - 1)) & 1:
            total += _popcount(S & _3ap_mask[x])
        x += 1
    return total


def construct():
    start = time.time()
    rng = random.Random(12345)
    best_S = 0
    best_count = 0

    def try_hillclimb(S):
        nonlocal best_S, best_count
        count = _count_3aps(S)
        # local search
        for _ in range(20000):
            if time.time() > _DEADLINE:
                break
            improved = False
            # candidate moves: add, remove, swap
            # additions
            for x in range(1, _N + 1):
                bit = 1 << (x - 1)
                if S & bit:
                    continue
                if S & _4ap_mask[x]:
                    continue
                S2 = S | bit
                c2 = count + _popcount(S2 & _3ap_mask[x])
                if c2 > count:
                    S = S2
                    count = c2
                    improved = True
                    break
            if improved:
                continue
            # removals
            for x in range(1, _N + 1):
                bit = 1 << (x - 1)
                if not (S & bit):
                    continue
                c2 = count - _popcount(S & _3ap_mask[x])
                if c2 > count:
                    S = S & ~bit
                    count = c2
                    improved = True
                    break
            if improved:
                continue
            # swaps
            for x in range(1, _N + 1):
                bit = 1 << (x - 1)
                if S & bit:
                    continue
                if S & _4ap_mask[x]:
                    continue
                for y in range(1, _N + 1):
                    bit2 = 1 << (y - 1)
                    if not (S & bit2):
                        continue
                    if (S & _4ap_mask[x]) & (S ^ bit2):
                        continue
                    S2 = (S & ~bit2) | bit
                    c2 = count - _popcount(S & _3ap_mask[y]) + _popcount(S2 & _3ap_mask[x])
                    if c2 > count:
                        S = S2
                        count = c2
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if count > best_count:
            best_count = count
            best_S = S

    # Phase 1: multi-ordering greedy constructions
    for it in range(200):
        if time.time() > _DEADLINE:
            break
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        S = 0
        for x in order:
            if S & _4ap_mask[x]:
                continue
            S |= 1 << (x - 1)
        try_hillclimb(S)

    # Phase 2: random restarts + hill climbing
    while time.time() < _DEADLINE:
        S = 0
        for x in range(1, _N + 1):
            if rng.random() < 0.5 and not (S & _4ap_mask[x]):
                S |= 1 << (x - 1)
        try_hillclimb(S)

    if best_S == 0:
        # fallback: original greedy
        S = 0
        for x in range(1, _N + 1):
            if not (S & _4ap_mask[x]):
                S |= 1 << (x - 1)
        best_S = S

    result = []
    x = 1
    while x <= _N:
        if (best_S >> (x - 1)) & 1:
            result.append(x)
        x += 1
    if not result:
        result = [1]
    return result


if __name__ == "__main__":
    a = construct()
    print(len(a), sum(bin(x).count("1") for x in a))
