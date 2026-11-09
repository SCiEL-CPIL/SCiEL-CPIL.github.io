"""Difference triangle set (7,5) – combined search with multi-start diversification,
aggressive row replacement, and targeted single-entry reduction."""
import time
import random
import math
import sys

_START = time.time()
_DEADLINE = _START + 145.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """Return frozenset of all positive differences in a row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        prev = -1
        for v in r:
            if v <= prev:
                return False
            prev = v
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


class DiffTracker:
    """Fast difference tracking using a boolean array for O(1) membership."""
    def __init__(self, max_diff):
        self.max_diff = max_diff
        self.used = bytearray(max_diff + 1)
        self.count = 0

    def add(self, diffs):
        for d in diffs:
            if d <= self.max_diff and not self.used[d]:
                self.used[d] = 1
                self.count += 1

    def remove(self, diffs):
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                self.used[d] = 0
                self.count -= 1

    def check(self, diffs):
        """Return True if none of the diffs conflict with current used set."""
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                return False
        return True

    def copy(self):
        nt = DiffTracker(self.max_diff)
        nt.used = bytearray(self.used)
        nt.count = self.count
        return nt


def _row_diffs_incremental(row, new_x):
    """Given a row (list) and a new candidate x, compute the diffs that would be added."""
    return {new_x - a for a in row}


def _interleaved_greedy(n_rows, k, max_mark, rnd=None, randomize_prob=0.0):
    """Interleaved greedy: extend the row with smallest current max at each step."""
    tracker = DiffTracker(max_mark + 1)
    rows = [[0] for _ in range(n_rows)]

    for _ in range(n_rows * k):
        if _dead():
            return None, None
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            if rnd is not None and rnd.random() < randomize_prob and x < max_mark:
                x += 1
                continue
            new_diffs = _row_diffs_incremental(row, x)
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None, None

    if all(len(r) == k + 1 for r in rows):
        return rows, tracker
    return None, None


def _sequential_greedy(n_rows, k, max_mark, rnd=None, randomize_prob=0.0):
    """Sequential greedy packing."""
    tracker = DiffTracker(max_mark + 1)
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            found = False
            while x <= max_mark:
                if rnd is not None and rnd.random() < randomize_prob and x < max_mark:
                    x += 1
                    continue
                new_diffs = _row_diffs_incremental(row, x)
                if tracker.check(new_diffs):
                    row.append(x)
                    tracker.add(new_diffs)
                    found = True
                    break
                x += 1
            if not found:
                return None
        rows.append(row)
    return rows


def _targeted_reduction(rows, n_rows, k, time_budget):
    """Systematically try to reduce each entry of each row from its current value downward."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    max_mark = best_scope

    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline:
        if _dead():
            break
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            for pi in range(len(row) - 1, 0, -1):  # From right to left
                old_val = row[pi]
                # Compute all other diffs
                other_diffs = set()
                for r2 in best:
                    if r2 is not row:
                        other_diffs |= _diffs_of_row(r2)
                # Also diffs within this row not involving position pi
                row_diffs_other = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if i != pi and j != pi:
                            row_diffs_other.add(row[j] - row[i])
                forbidden = other_diffs | row_diffs_other

                for new_val in range(old_val - 1, row[pi - 1], -1):
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    # Check new diffs involving pi
                    new_diffs_pi = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs_pi.add(new_val - row[j])
                            else:
                                new_diffs_pi.add(row[j] - new_val)
                    if new_diffs_pi & forbidden:
                        continue
                    # Valid!
                    row[pi] = new_val
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return best, best_scope


def _replace_row(rows, ri, used, k, max_mark, rnd, randomize_prob=0.1, attempts=15):
    """Try to replace row ri with a smaller-scope alternative."""
    old_diffs = _diffs_of_row(rows[ri])
    used_other = used - old_diffs
    old_scope = rows[ri][-1]

    for attempt in range(attempts):
        row = [0]
        local = set()
        x = 0
        success = True
        while len(row) < k + 1:
            x += 1
            if x >= old_scope:
                success = False
                break
            if rnd.random() < randomize_prob and x < old_scope - 1:
                x += 1
                if x >= old_scope:
                    success = False
                    break
            bad = False
            for a in row:
                dv = x - a
                if dv in used_other or dv in local:
                    bad = True
                    break
            if not bad:
                row.append(x)
                local.update({x - a for a in row[:-1]})
        if success and len(row) == k + 1:
            new_diffs = _diffs_of_row(row)
            if not (new_diffs & used_other):
                return row, used_other | new_diffs
    return None, None


def _multi_start_hill_climb(rows, n_rows, k, time_budget, rnd):
    """Multi-start hill-climbing with aggressive row replacement."""
    t_end = time.time() + time_budget
    best = [r[:] for r in rows]
    best_scope = _scope(best)

    for start in range(5):
        if _dead() or time.time() >= t_end:
            break

        current = [r[:] for r in best]
        used = set()
        for r in current:
            used |= _diffs_of_row(r)
        cur_scope = _scope(current)

        no_improve = 0
        max_iters = 2000

        for iters in range(max_iters):
            if _dead() or time.time() >= t_end:
                break
            if no_improve > 100:
                break

            ri = rnd.randrange(n_rows)
            old_scope = current[ri][-1]

            new_row, new_used = _replace_row(
                current, ri, used, k, min(cur_scope, best_scope), rnd,
                randomize_prob=0.1 + 0.05 * start, attempts=10 + 5 * start
            )
            if new_row is not None:
                current[ri] = new_row
                used = new_used
                cur_scope = _scope(current)
                no_improve = 0
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best = [r[:] for r in current]
            else:
                no_improve += 1

        # Perturbation: replace 1-2 random rows to diversify
        if start < 4:
            for _ in range(rnd.randint(1, 2)):
                ri = rnd.randrange(n_rows)
                new_row, new_used = _replace_row(
                    current, ri, used, k, best_scope + 5, rnd,
                    randomize_prob=0.3, attempts=20
                )
                if new_row is not None:
                    current[ri] = new_row
                    used = new_used

    return best, best_scope


def _simulated_annealing(rows, n_rows, k, time_budget, rnd):
    """Simulated annealing to reduce scope."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope

    used = set()
    for r in current:
        used |= _diffs_of_row(r)

    deadline = time.time() + time_budget
    T = 25.0
    cooling = 0.997

    while time.time() < deadline and T > 0.05:
        if _dead():
            break

        move_type = rnd.random()
        if move_type < 0.55:
            # Move type 1: Replace one row
            ri = rnd.randint(0, n_rows - 1)
            old_diffs = _diffs_of_row(current[ri])
            used_without = used - old_diffs
            max_mark = min(current_scope, best_scope)

            new_row = None
            for attempt in range(12):
                row = [0]
                local = set()
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                    if rnd.random() < 0.12 and x < max_mark - 1:
                        x += 1
                        if x > max_mark:
                            success = False
                            break
                    bad = False
                    for a in row:
                        dv = x - a
                        if dv in used_without or dv in local:
                            bad = True
                            break
                    if not bad:
                        row.append(x)
                        local.update({x - a for a in row[:-1]})
                if success and len(row) == k + 1:
                    new_row = row
                    break

            if new_row is not None:
                new_diffs = _diffs_of_row(new_row)
                new_scope = max(
                    max(r[-1] for r in current if r is not current[ri]),
                    new_row[-1]
                )
                delta = new_scope - current_scope
                if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                    used = used_without | new_diffs
                    current[ri] = new_row
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]

        elif move_type < 0.85:
            # Move type 2: Reduce a single entry
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            pi = rnd.randint(1, len(row) - 1)
            old_val = row[pi]
            if old_val > row[pi - 1] + 1:
                new_val = rnd.randint(row[pi - 1] + 1, old_val - 1)
                if pi < len(row) - 1 and new_val >= row[pi + 1]:
                    new_val = row[pi + 1] - 1
                    if new_val <= row[pi - 1]:
                        new_val = old_val
                if new_val != old_val:
                    new_row = row[:]
                    new_row[pi] = new_val
                    old_diffs = _diffs_of_row(row)
                    new_diffs = _diffs_of_row(new_row)
                    remaining = set()
                    for r2 in current:
                        if r2 is not row:
                            remaining |= _diffs_of_row(r2)
                    if not (new_diffs & remaining):
                        new_scope = max(
                            max(r[-1] for r in current if r is not row),
                            new_row[-1]
                        )
                        delta = new_scope - current_scope
                        if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                            used = (used - old_diffs) | new_diffs
                            current[ri] = new_row
                            current_scope = new_scope
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best = [r[:] for r in current]
        else:
            # Move type 3: Shift tail entries down
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            if len(row) > 2 and row[-1] > row[-2] + 1:
                shift = rnd.randint(1, min(3, row[-1] - row[-2] - 1))
                new_row = row[:]
                for j in range(len(new_row) - 1, 0, -1):
                    new_row[j] -= shift
                    if new_row[j] <= new_row[j-1]:
                        break
                else:
                    old_diffs = _diffs_of_row(row)
                    new_diffs = _diffs_of_row(new_row)
                    remaining = set()
                    for r2 in current:
                        if r2 is not row:
                            remaining |= _diffs_of_row(r2)
                    if not (new_diffs & remaining):
                        new_scope = max(
                            max(r[-1] for r in current if r is not row),
                            new_row[-1]
                        )
                        delta = new_scope - current_scope
                        if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                            used = (used - old_diffs) | new_diffs
                            current[ri] = new_row
                            current_scope = new_scope
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best = [r[:] for r in current]

        T *= cooling

    return best, best_scope


def _backtrack_fill(n_rows, k, max_mark, rnd, time_budget=5.0):
    """Backtracking search with MRV heuristic."""
    deadline = time.time() + time_budget
    rows = [None] * n_rows
    tracker = DiffTracker(max_mark + 1)

    def _fill_row(row_idx, row, tr, pos, max_mark):
        if time.time() > deadline:
            return False
        if pos == k:
            rows[row_idx] = row[:]
            return True
        start_x = row[-1] + 1
        for x in range(start_x, max_mark + 1):
            new_diffs = _row_diffs_incremental(row, x)
            if tr.check(new_diffs):
                row.append(x)
                tr.add(new_diffs)
                # Try to fill remaining rows
                if _fill_remaining(row_idx + 1, tr, max_mark):
                    return True
                tr.remove(new_diffs)
                row.pop()
        return False

    def _fill_remaining(idx, tr, max_mark):
        if time.time() > deadline:
            return False
        if idx == n_rows:
            return True
        row = [0]
        return _fill_row(idx, row, tr, 0, max_mark)

    # Try multiple random row orderings
    for attempt in range(100):
        if time.time() > deadline:
            break
        perm = list(range(n_rows))
        rnd.shuffle(perm)
        rows[:] = [None] * n_rows
        tracker = DiffTracker(max_mark + 1)

        def solve2(pos):
            if time.time() > deadline:
                return False
            if pos == n_rows:
                return True
            orig_idx = perm[pos]
            row = [0]
            if _fill_row2(orig_idx, row, tracker, 0, max_mark, pos):
                return True
            return False

        def _fill_row2(row_idx, row, tr, pos, max_mark, solve_pos):
            if time.time() > deadline:
                return False
            if pos == k:
                rows[row_idx] = row[:]
                return solve2(solve_pos + 1)
            start_x = row[-1] + 1
            for x in range(start_x, max_mark + 1):
                new_diffs = _row_diffs_incremental(row, x)
                if tr.check(new_diffs):
                    row.append(x)
                    tr.add(new_diffs)
                    if _fill_row2(row_idx, row, tr, pos + 1, max_mark, solve_pos):
                        return True
                    tr.remove(new_diffs)
                    row.pop()
            return False

        if solve2(0):
            return rows

    return None


def construct(n_rows, k):
    rnd = random.Random(12345)
    best_rows = None
    best_scope = float('inf')

    lb = n_rows * k * (k + 1) // 2

    # Phase 1: Multiple initializations to find a good starting point
    init_max_mark = max(200, lb + 20)

    # Interleaved greedy
    rows, _ = _interleaved_greedy(n_rows, k, init_max_mark)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)

    if _dead():
        return best_rows

    # Sequential greedy
    rows2 = _sequential_greedy(n_rows, k, init_max_mark)
    if rows2 is not None:
        sc = _scope(rows2)
        if sc < best_scope:
            best_rows = rows2
            best_scope = sc

    if _dead():
        return best_rows

    # Randomized interleaved greedy (multiple starts)
    for seed in range(5):
        if _dead():
            break
        rnd_tmp = random.Random(seed * 7919 + 13)
        rows3, _ = _interleaved_greedy(n_rows, k, init_max_mark, rnd_tmp, randomize_prob=0.15)
        if rows3 is not None:
            sc = _scope(rows3)
            if sc < best_scope:
                best_rows = rows3
                best_scope = sc

    if _dead():
        return best_rows

    # Phase 2: Targeted reduction to tighten the solution
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 3.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 3: Multi-start hill climbing with row replacement
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 15.0
        if remaining > 5.0:
            best_rows, best_scope = _multi_start_hill_climb(
                best_rows, n_rows, k, remaining, rnd
            )

    if _dead():
        return best_rows

    # Phase 4: Simulated annealing for further improvement
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 5.0:
            best_rows, best_scope = _simulated_annealing(
                best_rows, n_rows, k, remaining, rnd
            )

    if _dead():
        return best_rows

    # Phase 5: Another round of targeted reduction
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 2.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 6: Backtracking search at current best scope
    if best_scope > lb:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 3.0:
            result = _backtrack_fill(n_rows, k, best_scope, rnd, time_budget=min(remaining, 10.0))
            if result is not None:
                sc = _scope(result)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = result

    if _dead():
        return best_rows

    # Phase 7: Final targeted reduction
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 2.0
        if remaining > 1.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        # Fallback: re-run greedy
        rows, _ = _interleaved_greedy(n_rows, k, 200)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
