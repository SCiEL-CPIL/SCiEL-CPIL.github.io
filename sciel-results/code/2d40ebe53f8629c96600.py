# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_forbidden(m=205, k=2):
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    return forbidden


def _build_bitmasks(m, forbidden):
    # conflict_mask[i] is a bitmask of all j such that (i-j) mod m or (j-i) mod m is in forbidden
    conflict = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict[i] = mask
    return conflict


def _greedy_from_order(order, conflict, m):
    selected_mask = 0
    count = 0
    for cand in order:
        bit = 1 << cand
        if selected_mask & conflict[cand]:
            continue
        selected_mask |= bit
        count += 1
    return selected_mask, count


def _try_swaps(selected_mask, conflict, m, max_rounds=2000, rng=None):
    # Try to improve by removing one element and adding one or more new elements
    # Simple 1-for-1 swap: remove s, add c if c is free w.r.t. (selected - {s})
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        selected_list = [i for i in range(m) if (selected_mask >> i) & 1]
        for s in selected_list:
            reduced = selected_mask & ~(1 << s)
            # candidates: nodes not in reduced and not conflicting with reduced
            free_mask = 0
            # Build set of nodes conflicting with reduced
            blocked = 0
            for r in range(m):
                if (reduced >> r) & 1:
                    blocked |= conflict[r]
            # free = not blocked and not in reduced
            free = (~blocked) & ((1 << m) - 1) & ~reduced
            # Try to add a free node that increases count (we removed 1, so adding 1 keeps same; need to add >1 or find better)
            # Actually 1-for-1 swap keeps count same. We want net gain.
            # So try: remove s, then greedily add as many free nodes as possible.
            # If added > 1, net gain.
            added = 0
            new_mask = reduced
            # greedily add from free
            for c in range(m):
                if (free >> c) & 1:
                    if new_mask & conflict[c]:
                        continue
                    new_mask |= (1 << c)
                    added += 1
            if added > 1:
                selected_mask = new_mask
                improved = True
                break
    return selected_mask


def _try_add_only(selected_mask, conflict, m):
    # Try to add any node that fits
    improved = True
    while improved:
        improved = False
        for c in range(m):
            if (selected_mask >> c) & 1:
                continue
            if selected_mask & conflict[c]:
                continue
            selected_mask |= (1 << c)
            improved = True
    return selected_mask


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    forbidden = _compute_forbidden(m, 2)
    conflict = _build_bitmasks(m, forbidden)

    # Full mask of all m nodes
    full_mask = (1 << m) - 1

    rng = random.Random(12345)

    best_mask = 0
    best_count = 0

    def evaluate(mask):
        nonlocal best_mask, best_count
        cnt = bin(mask).count('1')
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
        return cnt

    # Phase 1: many random greedy restarts
    restarts = 0
    while time.time() < deadline - 120.0:
        order = list(range(m))
        rng.shuffle(order)
        mask, cnt = _greedy_from_order(order, conflict, m)
        evaluate(mask)
        restarts += 1
        if restarts % 200 == 0 and time.time() > deadline - 120.0:
            break

    # Phase 2: local improvement on best and top candidates
    # Keep top few masks from restarts by re-running with improvement
    # We'll run swap-based improvement on the best mask and a few random ones
    candidates = [best_mask]
    for _ in range(10):
        order = list(range(m))
        rng.shuffle(order)
        mask, cnt = _greedy_from_order(order, conflict, m)
        candidates.append(mask)

    for cm in candidates:
        if time.time() > deadline - 60.0:
            break
        cm = _try_add_only(cm, conflict, m)
        cm = _try_swaps(cm, conflict, m, max_rounds=300, rng=rng)
        cm = _try_add_only(cm, conflict, m)
        evaluate(cm)

    # Phase 3: if time remains, do more restarts + improvement
    while time.time() < deadline - 30.0:
        order = list(range(m))
        rng.shuffle(order)
        mask, cnt = _greedy_from_order(order, conflict, m)
        mask = _try_add_only(mask, conflict, m)
        mask = _try_swaps(mask, conflict, m, max_rounds=100, rng=rng)
        mask = _try_add_only(mask, conflict, m)
        evaluate(mask)

    # Build result array
    result = np.array([i for i in range(m) if (best_mask >> i) & 1], dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
