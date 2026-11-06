"""Snake-in-the-box in Q_9: bitmask DFS backtracking with MRV/Warnsdorff + blocked-mask propagation."""
import time
import random

N = 9
NUM_VERTICES = 1 << N  # 512
_MASK_BITS = 1 << NUM_VERTICES

_NEIGHBORS = None
_NBR_MASK = None
_POP = None
_DEADLINE = None
_RNG = None


def _init_data():
    global _NEIGHBORS, _NBR_MASK, _POP, _DEADLINE, _RNG
    if _NEIGHBORS is not None:
        return
    _NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
    _NBR_MASK = [0] * NUM_VERTICES
    for v in range(NUM_VERTICES):
        m = 0
        lst = []
        for b in range(N):
            nb = v ^ (1 << b)
            lst.append(nb)
            m |= (1 << nb)
        _NEIGHBORS[v] = lst
        _NBR_MASK[v] = m
    _POP = [0] * (1 << N)
    for i in range(1, 1 << N):
        _POP[i] = _POP[i & (i - 1)] + 1
    _DEADLINE = time.time() + 115.0
    _RNG = random.Random(1234567)


def _is_valid_snake(path):
    if len(path) <= 1:
        return True
    seen = set()
    for i, v in enumerate(path):
        if v in seen:
            return False
        seen.add(v)
        if i > 0 and (v ^ path[i - 1]).bit_count() != 1:
            return False
        for j in range(i - 2, -1, -1):
            if (v ^ path[j]).bit_count() == 1:
                return False
    return True


def _dfs(start):
    """DFS backtracking with Warnsdorff ordering and blocked-mask propagation.
    Returns the best path found (list of vertices)."""
    _init_data()
    deadline = _DEADLINE
    rng = _RNG
    best_path = [start]
    best_len = 1

    path = [start]
    used_mask = 1 << start
    blocked = _NBR_MASK[start]

    def _search():
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        cur = path[-1]
        plen = len(path)

        # Gather valid candidates (unused and not adjacent to any used vertex other than cur)
        cands = []
        for nb in _NEIGHBORS[cur]:
            if (used_mask >> nb) & 1:
                continue
            if (blocked >> nb) & 1:
                continue
            cands.append(nb)

        if not cands:
            if plen > best_len:
                best_len = plen
                best_path = path[:]
            return

        # Prune: even using every remaining vertex can't beat best
        if plen + (NUM_VERTICES - _POP[used_mask]) <= best_len:
            if plen > best_len:
                best_len = plen
                best_path = path[:]
            return

        # Warnsdorff scoring: count onward options for each candidate
        scored = []
        for nb in cands:
            new_blocked = blocked | _NBR_MASK[nb]
            onward = 0
            for nb2 in _NEIGHBORS[nb]:
                if (used_mask >> nb2) & 1:
                    continue
                if (new_blocked >> nb2) & 1:
                    continue
                onward += 1
            scored.append((onward, nb))
        scored.sort()

        # Randomize ties for diversification
        i = 0
        while i < len(scored):
            j = i
            while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
                j += 1
            if j > i:
                rng.shuffle(scored[i:j + 1])
            i = j + 1

        for _, nb in scored:
            if time.time() > deadline:
                return
            path.append(nb)
            used_mask |= (1 << nb)
            blocked |= _NBR_MASK[nb]
            _search()
            path.pop()
            used_mask ^= (1 << nb)
            blocked ^= _NBR_MASK[nb]
            if time.time() > deadline:
                return
            if best_len >= NUM_VERTICES:
                return

    _search()
    return best_path


def _extend(path):
    """Greedy frontier extension of a path using Warnsdorff + blocked-mask."""
    _init_data()
    deadline = _DEADLINE
    cur_path = list(path)
    used_mask = 0
    for v in cur_path:
        used_mask |= (1 << v)
    blocked = 0
    for v in cur_path:
        blocked |= _NBR_MASK[v]
    while True:
        if time.time() > deadline:
            break
        cur = cur_path[-1]
        cands = []
        for nb in _NEIGHBORS[cur]:
            if (used_mask >> nb) & 1:
                continue
            if (blocked >> nb) & 1:
                continue
            new_blocked = blocked | _NBR_MASK[nb]
            onward = 0
            for nb2 in _NEIGHBORS[nb]:
                if (used_mask >> nb2) & 1:
                    continue
                if (new_blocked >> nb2) & 1:
                    continue
                onward += 1
            cands.append((onward, nb))
        if not cands:
            break
        cands.sort()
        nb = cands[0][1]
        cur_path.append(nb)
        used_mask |= (1 << nb)
        blocked |= _NBR_MASK[nb]
    return cur_path


def construct():
    _init_data()
    deadline = _DEADLINE
    rng = _RNG

    best_path = None
    best_len = 0

    def _consider(p):
        nonlocal best_path, best_len
        if p and len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Deterministic first start
    _consider(_dfs(0))

    # Diversified randomized restarts
    starts = list(range(1, NUM_VERTICES))
    rng.shuffle(starts)
    for sv in starts:
        if time.time() > deadline - 3.0:
            break
        p = _dfs(sv)
        _consider(p)
        if best_len >= NUM_VERTICES:
            break

    # Extend best frontier if time remains
    if best_path is not None and time.time() < deadline - 1.0:
        e = _extend(best_path)
        _consider(e)

    if best_path is None:
        best_path = [0]

    if not _is_valid_snake(best_path):
        # Fallback simple greedy
        p = [0]
        used = {0}
        while True:
            cur = p[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for x in p[:-1]:
                    if (v ^ x).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            p.append(nxt)
            used.add(nxt)
        best_path = p

    return best_path
