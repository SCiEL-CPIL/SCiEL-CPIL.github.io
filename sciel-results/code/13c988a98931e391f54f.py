import time
import random
import sys

# EVOLVE-BLOCK-START

# ---------- Fast difference bookkeeping using boolean arrays ----------

class DiffTracker:
    """Tracks used positive differences up to a max value via a byte array.
    O(1) membership and O(1) add/remove of a single difference.
    """
    __slots__ = ('cap', 'used')

    def __init__(self, cap):
        self.cap = cap
        self.used = bytearray(cap + 1)  # index d -> 1 if used

    def is_free(self, d):
        return d <= self.cap and self.used[d] == 0

    def check_all(self, diffs):
        """Return True if all diffs in iterable are free."""
        u = self.used
        cap = self.cap
        for d in diffs:
            if d > cap or u[d]:
                return False
        return True

    def add(self, d):
        self.used[d] = 1

    def remove(self, d):
        self.used[d] = 0

    def add_many(self, diffs):
        u = self.used
        for d in diffs:
            u[d] = 1

    def remove_many(self, diffs):
        u = self.used
        for d in diffs:
            u[d] = 0

    def snapshot(self):
        return bytearray(self.used)

    def restore(self, snap):
        self.used = snap


def row_diffs(row):
    """List of all pairwise differences of a row (length k+1)."""
    out = []
    n = len(row)
    for i in range(n):
        a = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - a)
    return out


def row_diffs_set(row):
    return set(row_diffs(row))


def is_row_sidon(row):
    """All pairwise differences within the row distinct."""
    n = len(row)
    seen = set()
    for i in range(n):
        a = row[i]
        for j in range(i + 1, n):
            d = row[j] - a
            if d in seen:
                return False
            seen.add(d)
    return True


def max_entry(rows):
    return max(max(r) for r in rows)


def is_valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for d in row_diffs(row):
            if d in used:
                return False
            used.add(d)
    return True


# ---------- Greedy construction with conflict-awareness ----------

def greedy_construct(n_rows, k, cap, rng, strategy='smallest'):
    """
    Build rows greedily.
    strategy:
      'smallest'  : pick the smallest feasible next value.
      'spread'    : pick a feasible value that spreads out (largest gap friendly),
                    to reduce intra-row difference clustering.
      'random'    : pick a random feasible value.
    Returns (rows, used_tracker) or (None, None) if it cannot finish within cap.
    """
    tracker = DiffTracker(cap)
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        while len(row) < k + 1:
            found = False
            if strategy == 'smallest':
                order = range(last + 1, cap + 1)
            else:
                # build candidate list once
                cands = []
                for v in range(last + 1, cap + 1):
                    diffs = [v - a for a in row]
                    # intra-row distinct
                    if len(set(diffs)) != len(diffs):
                        continue
                    if not tracker.check_all(diffs):
                        continue
                    cands.append(v)
                if not cands:
                    break
                if strategy == 'spread':
                    # prefer larger values to spread
                    v = cands[-1]
                else:  # random
                    v = cands[rng.randrange(len(cands))]
                diffs = [v - a for a in row]
                row.append(v)
                tracker.add_many(diffs)
                last = v
                found = True
                continue

            # smallest strategy inline
            for v in range(last + 1, cap + 1):
                diffs = [v - a for a in row]
                if len(set(diffs)) != len(diffs):
                    continue
                if not tracker.check_all(diffs):
                    continue
                row.append(v)
                tracker.add_many(diffs)
                last = v
                found = True
                break
            if not found:
                return None, None
        rows.append(row)
    return rows, tracker


# ---------- Row re-solve with backtracking (bitmask accelerated) ----------

def solve_row(row_prefix, tracker, k, cap, rng, max_attempts=1, mode='any'):
    """
    Try to complete row_prefix (list starting at 0) to length k+1 with all values <= cap,
    using only differences not in tracker.
    mode:
      'any'  : any completion.
      'tight': prefer smaller max (we still just find any, caller controls cap).
    Returns completed row or None.
    """
    if len(row_prefix) == k + 1:
        return row_prefix[:]

    last = row_prefix[-1]
    # gather candidates
    cands = []
    for v in range(last + 1, cap + 1):
        diffs = [v - a for a in row_prefix]
        if len(set(diffs)) != len(diffs):
            continue
        if not tracker.check_all(diffs):
            continue
        cands.append(v)

    if not cands:
        return None

    # heuristic ordering: for 'tight' prefer smaller; else random shuffle
    if mode == 'tight':
        cands.sort()
    else:
        rng.shuffle(cands)

    for v in cands:
        diffs = [v - a for a in row_prefix]
        for d in diffs:
            tracker.add(d)
        new_row = row_prefix + [v]
        res = solve_row(new_row, tracker, k, cap, rng, max_attempts, mode)
        for d in diffs:
            tracker.remove(d)
        if res is not None:
            return res
    return None


# ---------- Local search: row replacement ----------

def try_replace_row(rows, tracker, ri, k, cap, rng, target_cap=None):
    """
    Remove row ri, then try to re-solve it with values <= (target_cap or cap).
    On success, return (new_rows, new_tracker) else (None, None).
    """
    if target_cap is None:
        target_cap = cap
    old_row = rows[ri]
    old_diffs = row_diffs(old_row)
    # remove old diffs
    for d in old_diffs:
        tracker.remove(d)

    new_row = solve_row([0], tracker, k, target_cap, rng, mode='any')
    if new_row is None:
        # restore
        for d in old_diffs:
            tracker.add(d)
        return None, None

    for d in row_diffs(new_row):
        tracker.add(d)
    new_rows = rows[:]
    new_rows[ri] = new_row
    return new_rows, tracker


# ---------- Main construct ----------

def construct(n_rows, k):
    start = time.time()
    deadline = start + 143.0  # safety margin under 150s

    # counting lower bound
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs

    rng = random.Random(12345)

    # Initial solution via greedy at a generous cap
    # greedy 'smallest' tends to give a compact solution
    init_cap = lb + 4 * k + 20
    rows, tracker = greedy_construct(n_rows, k, init_cap, rng, 'smallest')
    if rows is None:
        # widen
        rows, tracker = greedy_construct(n_rows, k, lb + 10 * k + 40, rng, 'smallest')
    if rows is None:
        # last resort: very wide
        rows, tracker = greedy_construct(n_rows, k, lb + 20 * k + 80, rng, 'smallest')

    best = [r[:] for r in rows]
    best_val = max_entry(best)
    best_tracker = DiffTracker(best_val)
    for r in best:
        best_tracker.add_many(row_diffs(r))

    # ---- Phase A: multi-start greedy at the counting lower bound region ----
    # Try to find a solution with max <= current best_val using diverse strategies.
    strategies = ['smallest', 'spread', 'random']
    for strat in strategies:
        if time.time() > deadline:
            break
        for _ in range(8):
            if time.time() > deadline:
                break
            r2, t2 = greedy_construct(n_rows, k, best_val, rng, strat)
            if r2 is not None and is_valid(r2, n_rows, k):
                v2 = max_entry(r2)
                if v2 < best_val:
                    best = [x[:] for x in r2]
                    best_val = v2
                    best_tracker = DiffTracker(best_val)
                    for rr in best:
                        best_tracker.add_many(row_diffs(rr))

    # ---- Phase B: iterated local search (row replacement) ----
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_tracker = DiffTracker(cur_val)
    for r in cur:
        cur_tracker.add_many(row_diffs(r))

    no_improve = 0
    max_no_improve = 200000
    step = 0

    while time.time() < deadline and no_improve < max_no_improve:
        step += 1
        # pick a random row to replace
        ri = rng.randrange(n_rows)
        old_max = max(cur[ri])
        # try to tighten: target cap = min(cur_val, old_max - 1) but not below 0
        target = min(cur_val, old_max - 1)
        if target < k:  # need at least k to place k more elements after 0
            target = cur_val

        new_rows, new_tracker = try_replace_row(cur, cur_tracker, ri, k, cur_val, rng, target_cap=target)
        if new_rows is not None:
            new_val = max_entry(new_rows)
            if new_val < cur_val:
                cur = new_rows
                cur_tracker = new_tracker
                cur_val = new_val
                best = [x[:] for x in cur]
                best_val = cur_val
                best_tracker = DiffTracker(best_val)
                for rr in best:
                    best_tracker.add_many(row_diffs(rr))
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Occasionally do a "kick": replace a random row without tightening,
        # to escape local optima (accept equal max).
        if no_improve % 500 == 0:
            ri = rng.randrange(n_rows)
            new_rows, new_tracker = try_replace_row(cur, cur_tracker, ri, k, cur_val, rng, target_cap=cur_val)
            if new_rows is not None:
                cur = new_rows
                cur_tracker = new_tracker
                # cur_val unchanged
                no_improve = min(no_improve, 200)

    # ---- Final verification ----
    if not is_valid(best, n_rows, k):
        # rebuild a guaranteed-valid greedy
        wide = best_val + 50
        rows, _ = greedy_construct(n_rows, k, wide, random.Random(7), 'smallest')
        if rows is not None:
            best = rows

    return best

# EVOLVE-BLOCK-END
