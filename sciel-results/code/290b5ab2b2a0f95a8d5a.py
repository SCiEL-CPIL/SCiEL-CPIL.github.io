"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_N = 256
_SUM_MAX = 2 * _N  # 512

def _build():
    # Perfect squares in the achievable sum range [2, 512]
    squares = set()
    k = 2
    while k * k <= _SUM_MAX:
        squares.add(k * k)
        k += 1

    # Precompute self-conflicts: 2a is a perfect square -> a cannot appear (diagonal)
    self_conflict = [False] * (_N + 1)
    for a in range(1, _N + 1):
        if 2 * a in squares:
            self_conflict[a] = True

    valid = [x for x in range(1, _N + 1) if not self_conflict[x]]

    # Conflict matrix among valid numbers (symmetric, a<b)
    conflict = [[0] * (_N + 1) for _ in range(_N + 1)]
    for i in range(len(valid)):
        a = valid[i]
        for j in range(i + 1, len(valid)):
            b = valid[j]
            if (a + b) in squares:
                conflict[a][b] = 1
                conflict[b][a] = 1

    # Degree of each valid number within the valid set
    degree = [0] * (_N + 1)
    for i in range(len(valid)):
        a = valid[i]
        for j in range(i + 1, len(valid)):
            b = valid[j]
            if conflict[a][b]:
                degree[a] += 1
                degree[b] += 1

    return squares, self_conflict, valid, conflict, degree

_SQUARES, _SELF, _VALID, _CONFLICT, _DEGREE = _build()

def _is_valid(s):
    for a in s:
        if _SELF[a]:
            return False
        for b in s:
            if a <= b and (a + b) in _SQUARES:
                return False
    return True

def _greedy_init():
    # Minimum-degree greedy: pick the valid number with fewest conflicts with the
    # already-chosen set; ties broken by smaller global degree, then value.
    chosen = set()
    for _ in range(len(_VALID)):
        best = None
        best_key = None
        for x in _VALID:
            if x in chosen:
                continue
            c = 0
            for y in chosen:
                if _CONFLICT[x][y]:
                    c += 1
            key = (c, _DEGREE[x], x)
            if best_key is None or key < best_key:
                best_key = key
                best = x
        chosen.add(best)
    return chosen

def _local_search(current, deadline):
    current = set(current)
    size = len(current)
    excluded = set(_VALID) - current
    in_set = [False] * (_N + 1)
    for x in current:
        in_set[x] = True

    improved = True
    while improved and (time.time() < deadline):
        improved = False

        # --- 1-0: add any excluded element with no conflict in the set ---
        for x in list(excluded):
            if time.time() >= deadline:
                break
            ok = True
            for y in current:
                if _CONFLICT[x][y]:
                    ok = False
                    break
            if ok:
                current.add(x)
                excluded.discard(x)
                in_set[x] = True
                size += 1
                improved = True

        # --- 1-1: swap y (in set) for x (excluded) when x only conflicts with y ---
        for x in list(excluded):
            if time.time() >= deadline:
                break
            # collect conflicts of x within current
            cf = []
            for y in current:
                if _CONFLICT[x][y]:
                    cf.append(y)
                    if len(cf) > 1:
                        break
            if len(cf) == 1:
                y = cf[0]
                # removing y and adding x keeps size; only worthwhile if it enables
                # further additions. Do the swap and let the next 1-0 pass exploit it.
                current.discard(y)
                excluded.add(y)
                in_set[y] = False
                current.add(x)
                excluded.discard(x)
                in_set[x] = True
                improved = True

        # --- 1-2: remove one y, add up to two non-conflicting excluded elements ---
        for y in list(current):
            if time.time() >= deadline:
                break
            added = []
            for x in excluded:
                if len(added) == 2:
                    break
                # x must not conflict with (current - {y}) nor with added
                ok = True
                for z in current:
                    if z != y and _CONFLICT[x][z]:
                        ok = False
                        break
                if ok:
                    for z in added:
                        if _CONFLICT[x][z]:
                            ok = False
                            break
                if ok:
                    added.append(x)
            if len(added) == 2:
                current.discard(y)
                excluded.add(y)
                in_set[y] = False
                for x in added:
                    current.add(x)
                    excluded.discard(x)
                    in_set[x] = True
                size += 1
                improved = True

    return current

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before the 90s kill

    # Deterministic, fast seed
    current = _greedy_init()

    if not _is_valid(current):
        # Defensive rebuild
        current = set()
        for x in sorted(_VALID, key=lambda v: (_DEGREE[v], v)):
            if all(not _CONFLICT[x][y] for y in current):
                current.add(x)

    # Time-bounded local search
    current = _local_search(current, deadline)

    # Final guard: ensure validity and nonemptiness
    if not _is_valid(current):
        current = _greedy_init()

    if not current:
        current = {1}

    return sorted(current)
# EVOLVE-BLOCK-END
