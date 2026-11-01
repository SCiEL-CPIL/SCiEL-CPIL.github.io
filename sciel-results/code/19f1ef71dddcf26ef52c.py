import time
import itertools
import numpy as np
import random

# EVOLVE-BLOCK-START

def _build_balls(q, n, radius, deadline):
    """
    Build the ball sets. For radius 1, ball(c) = {c} U {c with one symbol changed}.
    Returns a list of Python sets (each of size 1 + n*(q-1)).
    """
    total = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    balls = []
    for c in range(total):
        b = {c}
        for i in range(n):
            old = (c // powers[i]) % q
            for nd in range(q):
                if nd != old:
                    b.add(c + (nd - old) * powers[i])
        balls.append(b)
        if time.time() > deadline - 20:
            # Should not happen for n=8 (6561 balls, fast), but guard.
            break
    return balls

def _bits_to_set(bits, total):
    """Convert a 64-bit integer bitmask to a set of indices (for small masks)."""
    s = set()
    i = 0
    while bits:
        if bits & 1:
            s.add(i)
        bits >>= 1
        i += 1
    return s

def _greedy_upper_bound(q, n, radius, balls, deadline):
    """
    Greedy set cover: repeatedly pick the ball covering the most uncovered elements.
    Returns (number_selected, list_of_selected_indices).
    """
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    remaining = np.array([len(b) for b in balls], dtype=np.int32)
    selected = []

    # Precompute inverse: for each element y, list of balls containing y
    # (needed? For greedy we just scan remaining; but updating remaining needs inv)
    inv = [[] for _ in range(total)]
    for i, b in enumerate(balls):
        for y in b:
            inv[y].append(i)

    while True:
        if time.time() > deadline - 10:
            break
        idx = int(np.argmax(remaining))
        if remaining[idx] == 0:
            break
        selected.append(idx)
        for y in balls[idx]:
            if not covered[y]:
                covered[y] = True
                for i in inv[y]:
                    remaining[i] -= 1
        if covered.all():
            break
    return len(selected), selected

def _branch_and_bound(q, n, radius, balls, deadline, initial_ub):
    """
    Exact minimum set cover via branch-and-bound.
    State: uncovered (np bool array), remaining (np int array of |ball_i \ covered|).
    Prune if len(selected) + lower_bound >= best.
    """
    total = q ** n
    best = [initial_ub]
    best_set = [None]
    node_count = [0]

    # Precompute inv for fast updates
    inv = [[] for _ in range(total)]
    for i, b in enumerate(balls):
        for y in b:
            inv[y].append(i)

    # Initial state: nothing selected, all uncovered
    uncovered = np.ones(total, dtype=bool)
    remaining = np.array([len(b) for b in balls], dtype=np.int32)
    selected = []

    def lower_bound(uncovered_count, remaining):
        if uncovered_count == 0:
            return 0
        max_cov = int(remaining.max())
        if max_cov == 0:
            return 10**9  # infeasible
        return int(np.ceil(uncovered_count / max_cov))

    def update_after_add(c, uncovered, remaining):
        """Return (new_uncovered, new_remaining, changed_count) after adding ball c."""
        new_uncovered = uncovered.copy()
        new_remaining = remaining.copy()
        changed = 0
        for y in balls[c]:
            if new_uncovered[y]:
                new_uncovered[y] = False
                changed += 1
                for i in inv[y]:
                    new_remaining[i] -= 1
        return new_uncovered, new_remaining, changed

    def update_after_remove(c, uncovered, remaining):
        """Return (new_uncovered, new_remaining, changed_count) after removing ball c."""
        new_uncovered = uncovered.copy()
        new_remaining = remaining.copy()
        changed = 0
        for y in balls[c]:
            if not new_uncovered[y]:
                # Only if this ball was the sole coverer? We track cover counts separately.
                # Actually, to remove we need to know if y becomes uncovered.
                # We need cover_count. Let's maintain cover_count in the state.
                pass
        # This is getting complex. Let's maintain cover_count explicitly.
        return new_uncovered, new_remaining, changed

    # Let's restructure to maintain cover_count for easy removal.
    cover_count = np.zeros(total, dtype=np.int32)

    def lb(uncovered_count, remaining):
        if uncovered_count == 0:
            return 0
        m = int(remaining.max())
        if m == 0:
            return 10**9
        return int(np.ceil(uncovered_count / m))

    def search(uncovered, remaining, cover_count, selected):
        node_count[0] += 1
        if time.time() > deadline - 2:
            return

        unc_count = int(np.sum(uncovered))
        if unc_count == 0:
            if len(selected) < best[0]:
                best[0] = len(selected)
                best_set[0] = list(selected)
            return

        # Prune
        if len(selected) + lb(unc_count, remaining) >= best[0]:
            return

        # Candidate selection: choose an uncovered element y with fewest covering balls
        # (branching heuristic)
        # Find uncovered elements
        # To avoid O(total) scan every node, we could track, but for now scan.
        # Optimization: pick y with min len(inv[y]) among uncovered.
        # This is O(total * avg_deg) which is 6561 * 25 ~ 164k per node. Might be slow.
        # Let's try a simpler heuristic: pick the uncovered element with the smallest
        # number of available (remaining > 0) covering balls.
        
        # Actually, a standard heuristic is to branch on the element with the fewest
        # candidates. Let's find that.
        min_count = 10**9
        min_y = -1
        # Iterate only over uncovered? We have a bool array.
        # np.where(uncovered) gives indices.
        unc_indices = np.where(uncovered)[0]
        if len(unc_indices) == 0:
            return
            
        # For each, count how many balls in inv[y] have remaining > 0
        # This is expensive. Let's approximate with len(inv[y]) for now, or
        # just pick the first uncovered?
        # Let's pick the uncovered element with the smallest number of *available* covers.
        # To speed up, we can precompute for each y the list of balls.
        # Let's just pick a random uncovered element to keep it fast, or the one with min static degree.
        # Let's use static degree min.
        for y in unc_indices:
            # Count available covers
            cnt = 0
            for i in inv[y]:
                if remaining[i] > 0:
                    cnt += 1
            if cnt < min_count:
                min_count = cnt
                min_y = y
            if cnt == 1:
                break  # Can't get better than 1 (forced)

        if min_y == -1:
            return

        # Branch on min_y: must pick one of the balls covering min_y
        candidates = [i for i in inv[min_y] if remaining[i] > 0]
        
        # Order candidates by remaining coverage (descending) to find good solutions faster
        candidates.sort(key=lambda i: remaining[i], reverse=True)

        for c in candidates:
            # Add c
            new_cover_count = cover_count + 1  # vectorized? No, cover_count is array.
            # Let's do it manually to be safe and fast for small changes
            new_uncovered = uncovered.copy()
            new_remaining = remaining.copy()
            new_cover_count = cover_count.copy()
            
            for y in balls[c]:
                if new_uncovered[y]:
                    new_uncovered[y] = False
                    for i in inv[y]:
                        new_remaining[i] -= 1
                new_cover_count[y] += 1

            selected.append(c)
            search(new_uncovered, new_remaining, new_cover_count, selected)
            selected.pop()

    search(uncovered, remaining, cover_count, selected)
    return best[0], best_set[0]

def _int_to_digits(x, q, n, powers):
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    try:
        balls = _build_balls(q, n, radius, deadline)
        total = q ** n

        # 1. Get a good upper bound via Greedy
        greedy_size, greedy_set = _greedy_upper_bound(q, n, radius, balls, deadline)

        # 2. Try Branch and Bound to improve
        # Only if greedy is not too bad and we have time
        if time.time() < deadline - 30:
            bb_size, bb_set = _branch_and_bound(q, n, radius, balls, deadline, greedy_size)
            if bb_set is not None and bb_size < greedy_size:
                final_set = bb_set
            else:
                final_set = greedy_set
        else:
            final_set = greedy_set

        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q

        codewords = []
        for idx in final_set:
            codewords.append(_int_to_digits(idx, q, n, powers))

        return codewords

    except Exception as e:
        # Fallback: return a valid (though possibly large) cover
        # If anything fails, try a simple exhaustive-ish or random valid cover
        # Actually, if balls failed, we can't do much. But for n=8 it shouldn't fail.
        # Let's try a very simple greedy without precomputed balls if needed?
        # No, just re-raise or return empty? No, must be valid.
        # Let's assume _build_balls works.
        raise e

# EVOLVE-BLOCK-END
