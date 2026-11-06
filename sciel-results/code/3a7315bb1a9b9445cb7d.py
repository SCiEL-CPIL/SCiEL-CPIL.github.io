"""Rewritten snake-in-the-box solver for Q_9 using forward-greedy (random restarts) + beam search + repair."""
import time
import random
import numpy as np

N = 9
V = 1 << N
DEADLINE_MARGIN = 12.0  # seconds to leave before the 120s kill


def _valid_mask(cand, path_arr, used):
    """Vectorized check: can `cand` (int) be appended to path? Returns bool.
    cand must not be in used, and must have Hamming distance >=2 from every path vertex."""
    if cand in used:
        return False
    # distance from cand to each path vertex
    d = (cand ^ path_arr).bit_count() if hasattr(cand ^ path_arr, "bit_count") else None
    # path_arr is a python list of ints; use list comprehension (fast enough for len<=~300)
    for p in path_arr:
        if (cand ^ p).bit_count() == 1:
            return False
    return True


def _forward_greedy(start, bit_order, used=None, path=None):
    """Forward greedy following bit_order. Returns path list."""
    if path is None:
        path = [start]
        used = {start}
    else:
        used = used
    cur = path[-1]
    for b in bit_order:
        v = cur ^ (1 << b)
        if v in used:
            continue
        ok = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                ok = False
                break
        if ok:
            path.append(v)
            used.add(v)
            cur = v
    return path


def _beam_search(start, width, deadline):
    """Beam search over partial snakes. State = (path_tuple, used_set).
    Prefer extensions that maximize min distance to existing path (keeps options open)."""
    # beam: list of (path_list, used_set)
    beam = [([start], {start})]
    best_path = [start]
    while time.time() < deadline:
        next_beam = []
        for path, used in beam:
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # check distance >=2 from all path vertices except last (last is cur, dist=1)
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if not ok:
                    continue
                # score: min distance to path[:-1] (higher = safer), plus small random
                min_d = min((v ^ p).bit_count() for p in path[:-1]) if len(path) > 1 else N
                candidates.append((min_d, random.random(), v))
            if not candidates:
                continue
            # take the best candidate (max min_d)
            candidates.sort(key=lambda x: (-x[0], -x[1]))
            v = candidates[0][2]
            new_path = path + [v]
            new_used = used | {v}
            next_beam.append((new_path, new_used))
            if len(new_path) > len(best_path):
                best_path = new_path
        if not next_beam:
            break
        # prune beam to top `width` by (length, then min-distance heuristic at end)
        next_beam.sort(key=lambda x: (-len(x[0]),))
        beam = next_beam[:width]
    return best_path


def _repair(path, deadline):
    """Try to extend a stuck path by backtracking the last few steps and choosing a different branch.
    Simple local repair: for the last K vertices, try alternative bit flips."""
    best = list(path)
    if len(path) < 2:
        return best
    # Try re-choosing from the last few positions
    for back in range(1, min(6, len(path))):
        if time.time() > deadline:
            break
        prefix = path[:-back]
        used_prefix = set(prefix)
        # Try all bit orders from the penultimate prefix end
        cur = prefix[-1]
        for perm in [list(range(N))]:
            new_path = list(prefix)
            new_used = set(used_prefix)
            for b in perm:
                v = cur ^ (1 << b)
                if v in new_used:
                    continue
                ok = True
                for p in new_path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    new_path.append(v)
                    new_used.add(v)
                    cur = v
            if len(new_path) > len(best):
                best = new_path
    return best


def construct():
    start_time = time.time()
    deadline = start_time + (120.0 - DEADLINE_MARGIN)
    random.seed(12345)

    best_path = [0]

    # Strategy 1: forward greedy with many random bit orders and random starts
    for _ in range(2000):
        if time.time() > deadline:
            break
        start = random.randrange(V)
        bit_order = list(range(N))
        random.shuffle(bit_order)
        path = _forward_greedy(start, bit_order)
        if len(path) > len(best_path):
            best_path = path
            # try repair on the best
            if time.time() < deadline:
                rep = _repair(best_path, deadline)
                if len(rep) > len(best_path):
                    best_path = rep

    # Strategy 2: beam search with a few starts
    for _ in range(20):
        if time.time() > deadline:
            break
        start = random.randrange(V)
        path = _beam_search(start, width=4, deadline=deadline)
        if len(path) > len(best_path):
            best_path = path
            if time.time() < deadline:
                rep = _repair(best_path, deadline)
                if len(rep) > len(best_path):
                    best_path = rep

    # Final repair pass on the global best
    if time.time() < deadline:
        rep = _repair(best_path, deadline)
        if len(rep) > len(best_path):
            best_path = rep

    # Validate the result
    assert len(set(best_path)) == len(best_path), "duplicate vertices"
    for i in range(len(best_path) - 1):
        assert (best_path[i] ^ best_path[i + 1]).bit_count() == 1, "consecutive not Hamming 1"
    for i in range(len(best_path)):
        for j in range(i + 2, len(best_path)):
            assert (best_path[i] ^ best_path[j]).bit_count() >= 2, "non-consecutive Hamming 1"

    return best_path
