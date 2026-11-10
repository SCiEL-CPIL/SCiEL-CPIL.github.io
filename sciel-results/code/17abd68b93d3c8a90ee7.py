# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 580.0  # safety margin before 600s limit


def _compute_forbidden(m):
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    return forbidden


def _build_adjacency(m, forbidden):
    # adj[i] is a bitmask of nodes adjacent to i
    adj = np.zeros(m, dtype=np.uint64)
    # Use Python int for bitmask since m=205 > 64
    adj = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (j - i) % m
            if d in forbidden:
                adj[i] |= (1 << j)
    return adj


def _greedy_random_order(m, adj, forbidden, rng, start_order=None):
    """Greedy: process candidates in given order, add if compatible with all selected."""
    selected = set()
    if start_order is None:
        order = list(range(m))
        rng.shuffle(order)
    else:
        order = start_order
    for cand in order:
        ok = True
        for s in selected:
            d = (cand - s) % m
            if d in forbidden:
                ok = False
                break
        if ok:
            selected.add(cand)
    return selected


def _try_improve(m, adj, forbidden, selected, rng, max_rounds=200):
    """Local search: try 1-swap (remove one, add one) and pure additions."""
    selected = set(selected)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        cur_size = len(selected)
        # Try pure additions
        for cand in range(m):
            if cand in selected:
                continue
            ok = True
            for s in selected:
                d = (cand - s) % m
                if d in forbidden:
                    ok = False
                    break
            if ok:
                selected.add(cand)
                improved = True
                break
        if improved:
            continue
        # Try 1-swap: remove s, add cand
        sel_list = list(selected)
        rng.shuffle(sel_list)
        cand_list = list(range(m))
        rng.shuffle(cand_list)
        for s in sel_list:
            if len(selected) <= cur_size:
                break
            remaining = selected - {s}
            for cand in cand_list:
                if cand in remaining:
                    continue
                ok = True
                for r in remaining:
                    d = (cand - r) % m
                    if d in forbidden:
                        ok = False
                        break
                if ok:
                    selected = remaining | {cand}
                    improved = True
                    break
            if improved:
                break
    return selected


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    forbidden = _compute_forbidden(m)
    adj = _build_adjacency(m, forbidden)

    rng = random.Random(12345)
    best = set()

    # Multiple random restarts with greedy + local improvement
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > _DEADLINE:
            break
        sel = _greedy_random_order(m, adj, forbidden, rng)
        sel = _try_improve(m, adj, forbidden, sel, rng, max_rounds=100)
        if len(sel) > len(best):
            best = set(sel)

    # Additional targeted restarts if time remains
    extra = 0
    while time.time() < _DEADLINE - 5.0 and extra < 500:
        extra += 1
        sel = _greedy_random_order(m, adj, forbidden, rng)
        sel = _try_improve(m, adj, forbidden, sel, rng, max_rounds=50)
        if len(sel) > len(best):
            best = set(sel)

    return np.array(sorted(best), dtype=np.int64)


# EVOLVE-BLOCK-END
