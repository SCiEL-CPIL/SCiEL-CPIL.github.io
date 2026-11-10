# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 630.0
_MARGIN = 20.0
_START = time.time()


def _build_forbidden(m):
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    return forbidden


def _greedy_seed(m, forbidden, order):
    selected = []
    for c in order:
        ok = True
        for s in selected:
            d1 = (c - s) % m
            d2 = (s - c) % m
            if d1 in forbidden or d2 in forbidden:
                ok = False
                break
        if ok:
            selected.append(c)
    return selected


def _try_add(sel, c, m, forbidden):
    for s in sel:
        d1 = (c - s) % m
        d2 = (s - c) % m
        if d1 in forbidden or d2 in forbidden:
            return False
    return True


def _local_search(sel, m, forbidden, rng, deadline, max_iters):
    sel = list(sel)
    best = sel
    best_len = len(sel)
    it = 0
    while it < max_iters:
        it += 1
        if time.time() > deadline:
            break
        # candidate pool: elements not in set
        in_set = set(sel)
        pool = [x for x in range(m) if x not in in_set]
        # try random add
        if pool:
            c = rng.choice(pool)
            if _try_add(sel, c, m, forbidden):
                sel.append(c)
                if len(sel) > best_len:
                    best_len = len(sel)
                    best = list(sel)
                continue
        # try swap: remove r, add c
        r = rng.choice(sel)
        if pool:
            c = rng.choice(pool)
            rest = [s for s in sel if s != r]
            ok = _try_add(rest, c, m, forbidden)
            if ok:
                # size unchanged, but may enable future adds; accept with prob
                if rng.random() < 0.5:
                    sel = rest + [c]
        # occasional remove to escape (only if not improving)
        if rng.random() < 0.1:
            r = rng.choice(sel)
            sel = [s for s in sel if s != r]
    return best


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    forbidden = _build_forbidden(m)
    deadline = _START + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    best = []
    best_len = 0

    # deterministic baseline
    base = _greedy_seed(m, forbidden, list(range(m)))
    best = base
    best_len = len(base)

    # randomized restarts with different orderings
    iters = 0
    while time.time() < deadline:
        iters += 1
        order = list(range(m))
        rng.shuffle(order)
        seed = _greedy_seed(m, forbidden, order)
        if len(seed) > best_len:
            best_len = len(seed)
            best = seed
        # local search from this seed
        improved = _local_search(seed, m, forbidden, rng, deadline, max_iters=2000)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved

    return np.array(sorted(best), dtype=np.int64)


# EVOLVE-BLOCK-END
