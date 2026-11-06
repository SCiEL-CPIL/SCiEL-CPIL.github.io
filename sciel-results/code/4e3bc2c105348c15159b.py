"""Snake-in-the-box in Q_9: greedy seed + fail-first backtracking extension."""
import time
import random
import numpy as np

N = 9
FULL = (1 << N) - 1
DEADLINE = 110.0  # safety margin vs 150s wall-clock kill

# Precompute single-bit masks
MASKS = [1 << b for b in range(N)]
# bit_count lookup for all 512 values
BC = np.fromiter((v.bit_count() for v in range(1 << N)), dtype=np.uint8, count=1 << N)


def _is_valid_extend(v, path, used, last):
    """v is a valid next vertex: unused and differs >=2 from all but the last vertex."""
    if v in used:
        return False
    # compare against all vertices except the last (which differs by 1 by construction)
    # vectorized: compute popcount of (v ^ path[:-1])
    if len(path) >= 2:
        arr = np.array(path[:-1], dtype=np.uint16)
        diffs = BC[(v ^ arr).astype(np.intp)]
        if np.any(diffs == 1):
            return False
    return True


def _count_options(v, used, path, last):
    """Number of valid extensions from v (for fail-first ordering)."""
    cnt = 0
    for m in MASKS:
        w = v ^ m
        if w in used:
            continue
        if len(path) >= 2:
            arr = np.array(path[:-1], dtype=np.uint16)
            diffs = BC[(w ^ arr).astype(np.intp)]
            if np.any(diffs == 1):
                continue
        # also w must not equal last? last is in path[:-1]? No: last = path[-1], path[:-1] excludes it.
        # w differing by 1 from last is fine (that's the new consecutive edge). But w must differ >=2 from path[:-1].
        cnt += 1
    return cnt


def _greedy_extend(path, used, deadline, bit_order):
    """Greedy: keep extending with first valid candidate in bit_order. Returns (path, used)."""
    while time.time() < deadline:
        cur = path[-1]
        nxt = None
        for b in bit_order:
            v = cur ^ MASKS[b]
            if _is_valid_extend(v, path, used, cur):
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path, used


def _fail_first_extend(path, used, deadline):
    """DFS backtracking extension with fail-first candidate ordering.
    Returns the best (longest) path found within deadline.
    """
    best_path = list(path)
    best_len = len(path)

    def dfs(path, used):
        nonlocal best_path, best_len
        if time.time() >= deadline:
            return
        cur = path[-1]
        # gather valid candidates
        cands = []
        for b in range(N):
            v = cur ^ MASKS[b]
            if _is_valid_extend(v, path, used, cur):
                cands.append(v)
        if not cands:
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            return
        # fail-first: order by fewest onward options
        cands.sort(key=lambda v: _count_options(v, used, path, cur))
        for v in cands:
            path.append(v)
            used.add(v)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            dfs(path, used)
            path.pop()
            used.discard(v)
            if time.time() >= deadline:
                return

    dfs(list(path), set(used))
    return best_path


def _validate(path):
    """Sanity check the induced-path property."""
    if len(path) < 2:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def construct():
    start = time.time()
    deadline = start + DEADLINE

    best = []
    best_len = 0

    # Deterministic greedy seed with a few bit orders
    bit_orders = [
        list(range(N)),
        list(range(N - 1, -1, -1)),
        [0, 2, 4, 6, 8, 1, 3, 5, 7],
        [8, 6, 4, 2, 0, 7, 5, 3, 1],
    ]
    for order in bit_orders:
        if time.time() >= deadline:
            break
        path, used = _greedy_extend([0], {0}, deadline, order)
        if len(path) > best_len:
            best_len = len(path)
            best = list(path)

    # Now try fail-first backtracking extension from the best greedy path
    if time.time() < deadline - 5:
        extended = _fail_first_extend(best, set(best), deadline - 5)
        if len(extended) > best_len:
            best_len = len(extended)
            best = list(extended)

    # A couple of randomized restarts if time remains
    rng = random.Random(12345)
    for _ in range(6):
        if time.time() >= deadline - 3:
            break
        order = list(range(N))
        rng.shuffle(order)
        path, used = _greedy_extend([0], {0}, deadline - 3, order)
        if len(path) > best_len:
            best_len = len(path)
            best = list(path)
        if time.time() < deadline - 3:
            extended = _fail_first_extend(best, set(best), deadline - 3)
            if len(extended) > best_len:
                best_len = len(extended)
                best = list(extended)

    # Final safety: ensure we return something valid; fall back to plain greedy if needed
    if not best or len(best) < 2:
        path, used = _greedy_extend([0], {0}, deadline, list(range(N)))
        best = path

    if not _validate(best):
        # last resort: plain greedy
        path, used = _greedy_extend([0], {0}, deadline, list(range(N)))
        best = path

    return best


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print("length:", len(res), "edges:", len(res) - 1, "valid:", _validate(res), "time:", time.time() - t0)
