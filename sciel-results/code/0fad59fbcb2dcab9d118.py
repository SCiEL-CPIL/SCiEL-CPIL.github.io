import time
import random
import heapq

N = 256


def _is_sidon(selected):
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _compute_sums(selected):
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums


def _is_maximal(selected, sums):
    selected_set = set(selected)
    for m in range(1, N + 1):
        if m in selected_set:
            continue
        if 2 * m in sums:
            continue
        blocked = False
        for a in selected:
            if m + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True


class _Tracker:
    """Incremental Sidon-sum tracker with fast maximal / coverage checks."""
    __slots__ = ("sums", "sel_set", "members", "size")

    def __init__(self):
        self.sums = set()
        self.sel_set = set()
        self.members = []
        self.size = 0

    def add(self, x):
        self.sums.add(2 * x)
        for a in self.members:
            self.sums.add(x + a)
        self.members.append(x)
        self.sel_set.add(x)
        self.size += 1

    def remove(self, x):
        self.sel_set.discard(x)
        self.members.remove(x)
        self.size -= 1
        for a in self.members:
            self.sums.discard(x + a)
        self.sums.discard(2 * x)

    def is_maximal(self):
        for m in range(1, N + 1):
            if m in self.sel_set:
                continue
            if 2 * m in self.sums:
                continue
            blocked = False
            for a in self.members:
                if m + a in self.sums:
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    def cover_count(self, x):
        """Number of missing elements whose addability is blocked by x."""
        cnt = 0
        for m in range(1, N + 1):
            if m in self.sel_set or m == x:
                continue
            if 2 * m in self.sums:
                continue
            # m is addable iff it is not blocked by some a in members
            blocked = False
            for a in self.members:
                if a == x:
                    continue
                if m + a in self.sums:
                    blocked = True
                    break
            if not blocked and (m + x) in self.sums:
                cnt += 1
        return cnt


def _greedy_random(deadline, rng):
    """Randomized greedy construction of a Sidon set (maximal by construction)."""
    elements = list(range(1, N + 1))
    rng.shuffle(elements)
    selected = []
    sums = set()
    for x in elements:
        if time.time() > deadline:
            break
        if 2 * x in sums:
            continue
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected


def _greedy_ascending():
    selected = []
    sums = set()
    for x in range(1, N + 1):
        if 2 * x in sums:
            continue
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected


def _greedy_descending():
    selected = []
    sums = set()
    for x in range(N, 0, -1):
        if 2 * x in sums:
            continue
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected


def _priority_reduce(start_set, deadline, rng):
    """
    Greedily remove elements (lowest coverage first) while preserving maximality.
    This finds a small subset that still covers every missing element.
    """
    tracker = _Tracker()
    for x in start_set:
        tracker.add(x)

    if not tracker.is_maximal():
        return None

    # Build priority queue of (cover_count, x)
    heap = []
    for x in tracker.members:
        c = tracker.cover_count(x)
        heapq.heappush(heap, (c, x))

    while time.time() < deadline and heap:
        c, x = heapq.heappop(heap)
        if x not in tracker.sel_set:
            continue
        # Recompute coverage for x (may have changed)
        cur_c = tracker.cover_count(x)
        if cur_c != c:
            heapq.heappush(heap, (cur_c, x))
            continue
        # Try removing x
        tracker.remove(x)
        if tracker.is_maximal():
            # keep removal; update coverage of remaining members
            for a in tracker.members:
                if a == x:
                    continue
                new_c = tracker.cover_count(a)
                heapq.heappush(heap, (new_c, a))
        else:
            # restore x
            tracker.add(x)
            # coverage of others unchanged by restore? Re-add x changed sums.
            # Push x back and refresh others.
            heapq.heappush(heap, (tracker.cover_count(x), x))
            for a in tracker.members:
                if a == x:
                    continue
                heapq.heappush(heap, (tracker.cover_count(a), a))

    return list(tracker.sel_set)


def _swap_improve(start_set, deadline, rng):
    """Try swaps (remove one, add another) to enable further reductions."""
    tracker = _Tracker()
    for x in start_set:
        tracker.add(x)

    current = list(start_set)
    improved = True
    iters = 0
    while improved and time.time() < deadline and iters < 60:
        improved = False
        iters += 1
        remove_candidates = list(current)
        rng.shuffle(remove_candidates)
        add_candidates = [x for x in range(1, N + 1) if x not in set(current)]
        rng.shuffle(add_candidates)
        for r in remove_candidates:
            if time.time() > deadline:
                break
            for a in add_candidates:
                if time.time() > deadline:
                    break
                # apply swap
                tracker.remove(r)
                tracker.add(a)
                # check Sidon (tracker.sums has duplicates only if collision)
                # Sidon check: no sum appears twice. We track as set, so need explicit.
                # Rebuild Sidon check from members
                cand = list(tracker.sel_set)
                if _is_sidon(cand):
                    if tracker.is_maximal():
                        current = cand
                        improved = True
                        break
                # revert
                tracker.remove(a)
                tracker.add(r)
            if improved:
                break
    return current


def _combined_local_search(start_set, deadline, rng):
    current = list(start_set)
    current = _priority_reduce(current, deadline, rng)
    if current is None:
        return list(start_set)
    if time.time() < deadline:
        current = _swap_improve(current, deadline, rng)
    if time.time() < deadline:
        current2 = _priority_reduce(current, deadline, rng)
        if current2 is not None:
            current = current2
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s

    best = None
    best_size = N + 1

    candidates = []

    c1 = _greedy_ascending()
    candidates.append(c1)
    c2 = _greedy_descending()
    candidates.append(c2)

    for seed in range(60):
        if time.time() > deadline:
            break
        rng = random.Random(1000 + seed)
        sel = _greedy_random(deadline, rng)
        if sel:
            candidates.append(sel)

    for cand in candidates:
        if time.time() > deadline:
            break
        if not cand:
            continue
        if not _is_sidon(cand):
            continue
        if not _is_maximal(cand, _compute_sums(cand)):
            # not maximal; make maximal by adding? greedy already maximal, skip
            continue
        rng = random.Random(hash(tuple(sorted(cand))) % (2 ** 31))
        minimized = _combined_local_search(cand, deadline, rng)
        if minimized and _is_sidon(minimized) and _is_maximal(minimized, _compute_sums(minimized)):
            if len(minimized) < best_size:
                best_size = len(minimized)
                best = list(minimized)

    if best is None:
        return _greedy_ascending()

    return best
