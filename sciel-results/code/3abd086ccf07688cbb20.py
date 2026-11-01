"""Difference triangle set (7,5) solver via randomized local search."""
import time
import random

_DEADLINE = 150.0 - 10.0  # safety margin before the 150s kill
_START = time.time()


def _used_diffs(rows):
    s = set()
    for r in rows:
        for i in range(len(r)):
            ai = r[i]
            for j in range(i + 1, len(r)):
                s.add(r[j] - ai)
    return s


def _conflict_count(rows, used):
    # number of difference values that appear in more than one occurrence
    cnt = {}
    for r in rows:
        for i in range(len(r)):
            ai = r[i]
            for j in range(i + 1, len(r)):
                d = r[j] - ai
                cnt[d] = cnt.get(d, 0) + 1
    c = 0
    for d, n in cnt.items():
        if n > 1:
            c += n - 1
    return c


def _init(rows, k, used):
    # greedy placement of a single row given a used-difference set
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        ok = True
        for a in row:
            if (x - a) in used:
                ok = False
                break
        if ok:
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
    return row


def _seed(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_init(rows, k, used))
    return rows, used


def _valid(rows):
    s = set()
    for r in rows:
        for i in range(len(r)):
            ai = r[i]
            for j in range(i + 1, len(r)):
                d = r[j] - ai
                if d in s:
                    return False
                s.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows)


def construct(n_rows, k):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 150.0 - 10.0

    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0]] * n_rows

    # deterministic seed
    rows, used = _seed(n_rows, k)
    best = [r[:] for r in rows]
    best_scope = _scope(best)

    if _valid(best):
        pass  # already valid
    # keep best_scope as current best upper bound

    rng = random.Random(20240711)

    total_marks = n_rows * k  # movable marks (exclude the leading 0 of each row)

    # Precompute candidate move targets lazily; we sample candidate values.
    # We'll maintain used-diff set incrementally by recomputing after each accepted move.

    # Local search: repeatedly pick a random mark and try to move it to a new value
    # that reduces conflicts / scope.

    # Work on a mutable copy
    cur = [r[:] for r in rows]
    cur_used = _used_diffs(cur)

    # We track the current best (valid) solution separately.
    # Objective: minimize conflicts first, then scope.
    cur_conf = _conflict_count(cur, cur_used)
    cur_scope = _scope(cur)

    # If seed is valid, record it.
    if cur_conf == 0:
        best = [r[:] for r in cur]
        best_scope = cur_scope

    # Candidate generation: for a mark, candidate values are around current value
    # plus a random offset, and small values (to shrink scope).
    def try_move():
        nonlocal cur, cur_used, cur_conf, cur_scope, best, best_scope
        ri = rng.randrange(n_rows)
        # choose a non-zero mark index within the row
        pos = rng.randrange(1, k + 1)
        old = cur[ri][pos]

        # Generate candidate values
        cands = set()
        # shrink: small values
        for v in range(1, min(old, 20) + 1):
            cands.add(v)
        # local perturbations
        span = max(cur_scope, 20)
        for _ in range(6):
            off = rng.randint(-span // 3, span // 3)
            v = old + off
            if v >= 1:
                cands.add(v)
        # random within scope
        for _ in range(4):
            cands.add(rng.randint(1, max(2, cur_scope)))
        # slightly beyond current scope to allow re-expansion
        for _ in range(2):
            cands.add(rng.randint(cur_scope, cur_scope + 20))

        cands.discard(old)
        if not cands:
            return

        # Evaluate best candidate by (conflicts, scope)
        best_cand = None
        best_key = None
        for v in cands:
            # temporarily set
            cur[ri][pos] = v
            # recompute used and conflicts (cheap enough for moderate k)
            new_used = _used_diffs(cur)
            new_conf = _conflict_count(cur, new_used)
            new_scope = _scope(cur)
            key = (new_conf, new_scope)
            if best_key is None or key < best_key:
                best_key = key
                best_cand = (v, new_used, new_conf, new_scope)
            # revert
            cur[ri][pos] = old

        if best_cand is None:
            return
        v, new_used, new_conf, new_scope = best_cand
        # accept if better, or equal-conf with random acceptance for exploration
        if new_conf < cur_conf:
            accept = True
        elif new_conf == cur_conf:
            if new_scope < cur_scope:
                accept = True
            else:
                # Metropolis-like: accept slightly worse scope with small prob
                accept = rng.random() < 0.15
        else:
            # allow occasional increase in conflicts to escape, but rarely
            accept = rng.random() < 0.02

        if accept:
            cur[ri][pos] = v
            cur_used = new_used
            cur_conf = new_conf
            cur_scope = new_scope
            if cur_conf == 0:
                s = _scope(cur)
                if s < best_scope:
                    best_scope = s
                    best = [r[:] for r in cur]
        else:
            # revert (already reverted)
            pass

    # Deadline-driven loop
    while True:
        if time.time() - _START > _DEADLINE:
            break
        # batch of moves
        for _ in range(200):
            if time.time() - _START > _DEADLINE:
                break
            try_move()
        # Occasionally restart from a fresh seed if stuck and time remains
        if time.time() - _START > _DEADLINE:
            break
        if cur_conf > 0 and rng.random() < 0.01:
            # restart with a new random seed
            rows2, used2 = _seed(n_rows, k)
            # randomize the seed a bit
            for _ in range(50):
                ri = rng.randrange(n_rows)
                pos = rng.randrange(1, k + 1)
                cur_scope_now = _scope(rows2)
                rows2[ri][pos] = rng.randint(1, max(2, cur_scope_now))
            cur = [r[:] for r in rows2]
            cur_used = _used_diffs(cur)
            cur_conf = _conflict_count(cur, cur_used)
            cur_scope = _scope(cur)
            if cur_conf == 0:
                s = _scope(cur)
                if s < best_scope:
                    best_scope = s
                    best = [r[:] for r in cur]

    # Ensure best is valid (it should be, but guard)
    if not _valid(best):
        # fallback: return a valid greedy seed
        rows, used = _seed(n_rows, k)
        best = rows
    return best
