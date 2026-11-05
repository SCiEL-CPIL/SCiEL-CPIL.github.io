import random
import time

N = 256
DEADLINE = 56.0  # safety margin under the 60 s wall-clock kill

# EVOLVE-BLOCK-START

def _precompute():
    # Conflict graph: for each element, the set of elements that share a sum with it.
    # s -> list of elements x with x <= s//2 and x+x or x+(s-x) == s.
    sums_elems = {}
    for x in range(1, N + 1):
        for y in range(x, N + 1):
            s = x + y
            sums_elems.setdefault(s, []).append((x, y))

    conflict = [set() for _ in range(N + 1)]
    for s, pairs in sums_elems.items():
        for i in range(len(pairs)):
            a1, b1 = pairs[i]
            for j in range(i + 1, len(pairs)):
                a2, b2 = pairs[j]
                for a in (a1, b1):
                    for b in (a2, b2):
                        if a != b:
                            conflict[a].add(b)
                            conflict[b].add(a)
    return conflict


CONFLICT = _precompute()


def _blocked_count(sel):
    """How many missing elements are destroyed by the Sidon set sel."""
    S = set(sel)
    used = set()
    for a in sel:
        used.add(2 * a)
        for b in sel:
            if b >= a:
                used.add(a + b)
    cnt = 0
    for x in range(1, N + 1):
        if x in S:
            continue
        if (2 * x) in used or any((x + a) in used for a in sel):
            cnt += 1
    return cnt


def _maximal_blocked(sel):
    """Return (is_maximal, missing_count)."""
    S = set(sel)
    used = set()
    for a in sel:
        used.add(2 * x) if False else used.add(2 * a)
        for b in sel:
            if b >= a:
                used.add(a + b)
    missing = 0
    for x in range(1, N + 1):
        if x in S:
            continue
        if (2 * x) in used or any((x + a) in used for a in sel):
            continue
        missing += 1
    return (missing == 0), missing


def _is_valid(sel):
    used = set()
    for a in sel:
        for b in sel:
            if b >= a:
                s = a + b
                if s in used:
                    return False
                used.add(s)
    return True


def _greedy_seed(rng):
    """Build a small Sidon set by greedy with random tie-breaking."""
    sel = []
    used = set()
    pool = list(range(1, N + 1))
    rng.shuffle(pool)
    # Bias toward large elements: sort with random jitter, larger first on average.
    for x in sorted(pool, key=lambda v: (-v, rng.random())):
        ok = True
        added = [2 * x]
        for a in sel:
            added.append(x + a)
        if len(set(added)) != len(added):
            continue
        if used.intersection(added):
            continue
        sel.append(x)
        used.update(added)
    return sel


def _local_search(sel, rng, iters):
    """Improve the blocking count via random add/remove/swap moves."""
    sel = list(sel)
    best_score = _blocked_count(sel)
    cur_score = best_score
    for _ in range(iters):
        if time.time() > DEADLINE:
            break
        move = rng.random()
        if move < 0.4:
            # Try adding a random missing element
            S = set(sel)
            cand = rng.randint(1, N)
            if cand in S:
                continue
            new_sel = sel + [cand]
            if not _is_valid(new_sel):
                continue
            ns = _blocked_count(new_sel)
            if ns >= cur_score:
                sel = new_sel
                cur_score = ns
                best_score = max(best_score, ns)
        elif move < 0.7:
            # Try removing a random element
            if len(sel) <= 1:
                continue
            i = rng.randrange(len(sel))
            new_sel = sel[:i] + sel[i + 1:]
            ns = _blocked_count(new_sel)
            if ns >= cur_score:
                sel = new_sel
                cur_score = ns
                best_score = max(best_score, ns)
        else:
            # Swap: remove one, add one
            if len(sel) < 2:
                continue
            i = rng.randrange(len(sel))
            removed = sel[i]
            base = sel[:i] + sel[i + 1:]
            cand = rng.randint(1, N)
            if cand in base or cand == removed:
                continue
            new_sel = base + [cand]
            if not _is_valid(new_sel):
                continue
            ns = _blocked_count(new_sel)
            if ns >= cur_score:
                sel = new_sel
                cur_score = ns
                best_score = max(best_score, ns)
    return sel, best_score


def _prune_nonessential(sel, rng):
    """Remove elements that are not needed for maximality (i.e., whose removal
    keeps the set maximal)."""
    sel = list(sel)
    changed = True
    while changed:
        if time.time() > DEADLINE:
            break
        changed = False
        for i in range(len(sel) - 1, -1, -1):
            trial = sel[:i] + sel[i + 1:]
            if not _is_valid(trial):
                continue
            is_max, _ = _maximal_blocked(trial)
            if is_max:
                sel = trial
                changed = True
                break
    return sel


def construct():
    start = time.time()
    rng = random.Random(12345)
    best_sel = None
    best_size = N + 1

    while time.time() < DEADLINE:
        # 1. seed a small Sidon core
        seed = _greedy_seed(rng)
        if not seed:
            continue
        # 2. local search to boost blocking
        cur, _ = _local_search(seed, rng, iters=40)
        # 3. ensure maximality; if not maximal, add blockers greedily
        S = set(cur)
        used = set()
        for a in cur:
            used.add(2 * a)
            for b in cur:
                if b >= a:
                    used.add(a + b)
        # add missing elements that are not yet blocked, in random order
        missing = [x for x in range(1, N + 1)
                   if x not in S and not ((2 * x) in used or any((x + a) in used for a in cur))]
        rng.shuffle(missing)
        for x in missing:
            new_sel = cur + [x]
            if _is_valid(new_sel):
                cur = new_sel
                S.add(x)
                used.add(2 * x)
                for a in cur:
                    if a != x:
                        used.add(x + a)
        # 4. prune non-essential elements
        cur = _prune_nonessential(cur, rng)
        if _is_valid(cur):
            is_max, _ = _maximal_blocked(cur)
            if is_max and len(cur) < best_size:
                best_size = len(cur)
                best_sel = list(cur)

    if best_sel is None:
        # fallback: deterministic ascending greedy (always valid & maximal)
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    return best_sel

# EVOLVE-BLOCK-END
