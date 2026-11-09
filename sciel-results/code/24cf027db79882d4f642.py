import time
import numpy as np

N = 9
SIZE = 1 << N
MASK = SIZE - 1
# Precompute bit masks for each dimension
BITM = [1 << b for b in range(N)]
# Precompute popcount for all vertices (bit_count is fast but a list is fastest)
PC = np.zeros(SIZE, dtype=np.int32)
for i in range(1, SIZE):
    PC[i] = PC[i >> 1] + (i & 1)

_DEADLINE = None
_T0 = None


def _time_left():
    if _T0 is None:
        return 120.0
    return _DEADLINE - time.monotonic()


def _is_valid_extension(v, path, used):
    """True if adding vertex v after current path keeps it an induced path."""
    # v must be adjacent to the last vertex
    if used[v]:
        return False
    # v must NOT be adjacent to any non-last vertex
    # path[-1] is the current end; check all of path[:-1]
    diffs = (path[:-1] ^ v)
    # if any has popcount 1 -> invalid
    return not np.any(PC[diffs] == 1)


def _score_end(cur, used, path, path_arr):
    """Warnsdorff: count how many valid neighbors cur has (future freedom).
    Lower = more constrained = better to commit now."""
    freedom = 0
    for b in range(N):
        w = cur ^ BITM[b]
        if used[w]:
            continue
        # w adjacent to cur (by construction). Check w vs path[:-1]
        if path_arr.size >= 1:
            diffs = (path_arr[:-1] ^ w)
            if np.any(PC[diffs] == 1):
                continue
        freedom += 1
    return freedom


def _grow_end(cur, used, path, path_arr, prefer_low):
    """Pick a valid neighbor of cur to append; returns (w, freedom) or (None, None)."""
    best = None
    best_freedom = None
    for b in range(N):
        w = cur ^ BITM[b]
        if used[w]:
            continue
        if path_arr.size >= 1:
            diffs = (path_arr[:-1] ^ w)
            if np.any(PC[diffs] == 1):
                continue
        # valid
        freedom = _score_end(w, used, path, path_arr)
        if best is None:
            best, best_freedom = w, freedom
        else:
            # Warnsdorff: pick min freedom; tie-break by bit index
            if freedom < best_freedom or (freedom == best_freedom and b < (best ^ cur).bit_length() - 1):
                best, best_freedom = w, freedom
    return best, best_freedom


def _build(start, end_limit):
    """Grow a snake from both ends. Returns list of vertices."""
    global _T0, _DEADLINE
    path = [start]
    used = np.zeros(SIZE, dtype=bool)
    used[start] = True
    path_arr = np.array([start], dtype=np.int64)

    left_cur = start
    right_cur = start
    # We grow from the right end primarily; alternate to keep both flexible.
    # Represent path as a list; left end = path[0], right end = path[-1].
    # Growing left means prepending; growing right means appending.

    grow_right = True
    while True:
        if _time_left() < 0.5:
            break
        if grow_right:
            w, _ = _grow_end(right_cur, used, path, path_arr)
            if w is None:
                # try left
                w2, _ = _grow_end(left_cur, used, path, path_arr)
                if w2 is None:
                    break
                # prepend
                path.insert(0, w2)
                used[w2] = True
                path_arr = np.concatenate(([w2], path_arr))
                left_cur = w2
            else:
                path.append(w)
                used[w] = True
                path_arr = np.concatenate((path_arr, [w]))
                right_cur = w
        else:
            w, _ = _grow_end(left_cur, used, path, path_arr)
            if w is None:
                w2, _ = _grow_end(right_cur, used, path, path_arr)
                if w2 is None:
                    break
                path.append(w2)
                used[w2] = True
                path_arr = np.concatenate((path_arr, [w2]))
                right_cur = w2
            else:
                path.insert(0, w)
                used[w] = True
                path_arr = np.concatenate(([w], path_arr))
                left_cur = w
        grow_right = not grow_right
    return path


def construct():
    global _T0, _DEADLINE
    _T0 = time.monotonic()
    _DEADLINE = _T0 + 115.0  # safety margin before 120s

    best = None
    # Try several seeds; keep the best snake found.
    seeds = [0, 1, 255, 128, 511, 7, 33, 100, 200, 400]
    for s in seeds:
        if _time_left() < 2.0:
            break
        p = _build(s, None)
        if best is None or len(p) > len(best):
            best = p
        # If we've hit a strong result, stop early to be safe.
        if len(p) >= 150:
            break

    # Fallback: pure greedy (original) in case something failed
    if best is None or len(best) < 104:
        path, used = [0], {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ BITM[b]
                if v in used:
                    continue
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        if best is None or len(path) > len(best):
            best = path

    return list(best)
