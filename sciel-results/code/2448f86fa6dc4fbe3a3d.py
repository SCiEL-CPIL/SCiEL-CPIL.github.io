"""Find a small-diameter Sidon set (B_2) containing 0 whose positive differences
include 1..30. Minimizes the largest element."""
import time
import random

_DEADLINE_FRACTION = 0.88
_WALL_LIMIT = 150.0


def _diffs_of(s):
    return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]


def _is_sidon(s):
    d = _diffs_of(s)
    return len(set(d)) == len(d)


def _covers(s):
    dset = set(_diffs_of(s))
    return all(d in dset for d in range(1, 31))


def _valid(s):
    if len(s) == 0 or 0 not in s:
        return False
    if len(s) > 32:
        return False
    if max(s) > 10**6:
        return False
    return _is_sidon(s) and _covers(s)


def _greedy_construct():
    """Fast greedy: place elements one at a time, trying to cover missing diffs."""
    a = [0]
    used = set()
    # We'll try to build up a set that covers 1..30
    # Strategy: for each needed difference d, try to add an element x such that
    # x - some_existing = d or some_existing - x = d
    # Also allow fresh elements.
    
    # Start with 0 and 1
    a = [0, 1]
    used = {1}
    
    needed = set(range(1, 31)) - {1}
    
    # Try to extend by adding elements
    max_val = 10**5  # upper bound for greedy
    for _ in range(100):
        if not needed:
            break
        best_x = None
        best_covers = 0
        best_new_diffs = None
        # Try candidate positions
        cands = set()
        for v in a:
            for d in list(needed):
                cands.add(v + d)
                cands.add(v - d)
        # Also add some random candidates
        for _ in range(50):
            cands.add(random.randint(1, max(2, 2 * max(a) + 10)))
        
        for x in sorted(cands):
            if x < 0 or x in a:
                continue
            trial = sorted(a + [x])
            if len(trial) > 32:
                continue
            nd = [abs(x - v) for v in a]
            if len(set(nd)) != len(nd):
                continue
            if any(nd_ in used for nd_ in nd):
                continue
            new_used = used | set(nd)
            new_needed = needed & new_used
            if len(new_needed) > best_covers:
                best_covers = len(new_needed)
                best_x = x
                best_new_diffs = nd
        if best_x is None:
            # Try a fresh pair
            y = 2 * max(a) + 5
            while y < max_val:
                trial = sorted(a + [y, y + 2])
                if len(trial) > 32:
                    break
                nd = []
                for v in a:
                    nd.append(y - v)
                    nd.append(y + 2 - v)
                nd.append(2)
                if len(set(nd)) == len(nd) and not any(nd_ in used for nd_ in nd):
                    best_x = (y, y + 2)
                    best_covers = 1
                    best_new_diffs = nd
                    break
                y += 1
            if best_x is None:
                break
        if isinstance(best_x, tuple):
            a = sorted(a + list(best_x))
            used.update(best_new_diffs)
        else:
            a = sorted(a + [best_x])
            used.update(best_new_diffs)
        needed = set(range(1, 31)) - used
    
    return a


def _local_search(a, rng, deadline, max_iters=20000):
    """Local search to minimize max element while maintaining validity."""
    cur = sorted(a)
    cur_max = max(cur)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if cur_max <= 30:
            break
        # Pick a random non-zero element to move
        idx = rng.randint(1, len(cur) - 1)
        old = cur[idx]
        if old == 1:
            continue
        # Try to move it to a smaller value
        new = rng.randint(1, old - 1)
        if new in cur:
            continue
        trial = cur[:idx] + [new] + cur[idx + 1:]
        trial = sorted(trial)
        if not _valid(trial):
            continue
        if max(trial) < cur_max:
            cur = trial
            cur_max = max(cur)
    return cur


def _backtrack_search(D, k, rng, deadline, iters=500):
    """Try to find a Sidon set of size k in [0, D] containing 0 covering 1..30."""
    for _ in range(iters):
        if time.time() > deadline:
            return None
        a = [0]
        used = set()
        order = list(range(1, 31))
        rng.shuffle(order)
        ok = True
        for d in order:
            if d in used:
                continue
            placed = False
            cands = []
            for v in a:
                for x in (v + d, v - d):
                    if 0 <= x <= D and x not in a:
                        cands.append(x)
            if len(a) < k:
                cands.append(rng.randint(1, D))
            rng.shuffle(cands)
            for x in cands:
                nd = [abs(x - v) for v in a]
                if len(set(nd)) != len(nd):
                    continue
                if any(nd_ in used for nd_ in nd):
                    continue
                a.append(x)
                used.update(nd)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(a) <= 32:
            a = sorted(set(a))
            if _valid(a):
                return a
    return None


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT * _DEADLINE_FRACTION
    
    best = None
    best_max = 10**6
    
    def consider(a):
        nonlocal best, best_max
        a = sorted(set(a))
        if _valid(a):
            m = max(a)
            if m < best_max:
                best_max = m
                best = list(a)
    
    rng = random.Random(42)
    
    # 1. Greedy baseline
    baseline = _greedy_construct()
    consider(baseline)
    
    # 2. Try known small constructions: perfect difference sets
    # A (v, k, lambda) difference set in Z_v gives a Sidon set of size k.
    # For lambda=1, k(k-1) = v-1, so v = k^2 - k + 1.
    # Known: k=7, v=43; k=11, v=111; k=13, v=163
    # The set of elements in [0, v-1] from a perfect difference set has all
    # differences 1..v-1 exactly once (cyclic). The linear version may not
    # cover all, but let's try.
    
    # Try to find small Sidon sets by direct construction
    # Known: {0, 1, 4, 10, 12, 17, 21, 23} is not Sidon (check: 1-0=1, 4-1=3, 10-4=6, ...)
    # Let me just use random search more aggressively.
    
    # 3. Randomized search over increasing diameters
    for D in range(30, 150):
        if time.time() > deadline:
            break
        for k in (8, 9, 10, 11, 12, 13, 14):
            if time.time() > deadline:
                break
            res = _backtrack_search(D, k, rng, deadline, iters=200)
            if res is not None:
                consider(res)
                if best_max <= D:
                    break
    
    # 4. Local search to shrink
    if best is not None:
        if time.time() < deadline:
            shrunk = _local_search(best, rng, deadline, max_iters=30000)
            consider(shrunk)
    
    # 5. More aggressive random restarts if time remains
    if best is not None and time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            # Perturb the best solution
            pert = list(best)
            for _ in range(5):
                idx = rng.randint(1, len(pert) - 1)
                old = pert[idx]
                if old > 1:
                    new = rng.randint(1, old)
                    if new not in pert:
                        pert[idx] = new
                        break
            consider(pert)
            if time.time() < deadline:
                shrunk = _local_search(pert, rng, deadline, max_iters=5000)
                consider(shrunk)
    
    if best is None:
        return baseline
    return best


if __name__ == "__main__":
    print(construct())
