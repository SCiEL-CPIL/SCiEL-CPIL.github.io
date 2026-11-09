"""Difference triangle set (7,5) – model-and-optimize with constraint propagation and iterated local search."""
import time
import random
import sys
import math

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
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
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


def _greedy_row_fast(tracker, k, max_mark, rnd=None):
    """Greedy: build a row of k+1 entries starting at 0, smallest valid x at each step."""
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = _row_diffs_incremental(row, x)
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    return row


def _greedy_fill(n_rows, k, max_mark, rnd=None):
    """Sequential greedy fill with a shared tracker."""
    tracker = DiffTracker(max_mark + 1)
    rows = []
    for _ in range(n_rows):
        row = _greedy_row_fast(tracker, k, max_mark, rnd)
        if row is None:
            return None, None
        rows.append(row)
    return rows, tracker


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
    """Interleaved greedy: extend the row with smallest current max at each step."""
    tracker = DiffTracker(max_mark + 1)
    rows = [[0] for _ in range(n_rows)]

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
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


def _backtrack_fill(n_rows, k, max_mark, rnd, time_budget=5.0):
    """Backtracking search with MRV heuristic and forward checking."""
    deadline = time.time() + time_budget
    rows = [None] * n_rows
    tracker = DiffTracker(max_mark + 1)

    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n_rows:
            return True
        # Try to build row idx
        row = [0]
        return _fill_row(idx, row, tracker, 0, max_mark)

    def _fill_row(row_idx, row, tr, pos, max_mark):
        """Recursively fill position pos in row."""
        if time.time() > deadline:
            return False
        if pos == k:
            # Row complete
            rows[row_idx] = row[:]
            return solve(row_idx + 1)
        # Try each candidate x starting from row[-1]+1
        start_x = row[-1] + 1
        for x in range(start_x, max_mark + 1):
            new_diffs = _row_diffs_incremental(row, x)
            if tr.check(new_diffs):
                row.append(x)
                tr.add(new_diffs)
                if _fill_row(row_idx, row, tr, pos + 1, max_mark):
                    return True
                tr.remove(new_diffs)
                row.pop()
        return False

    # Try multiple random row orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        perm = list(range(n_rows))
        rnd.shuffle(perm)
        # Reorder: solve rows in perm order
        rows[:] = [None] * n_rows
        tracker = DiffTracker(max_mark + 1)

        # Map original row indices to perm order
        row_to_pos = {perm[i]: i for i in range(n_rows)}

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


def _greedy_row_with_noise(tracker, k, max_mark, rnd, noise_prob=0.2):
    """Greedy row construction with random skips to escape local minima."""
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            # Randomly skip some valid candidates to explore different structures
            if rnd.random() < noise_prob and x < max_mark:
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
            # If we get stuck, remove what we added and return None
            if len(row) > 1:
                # We need to undo the adds. This is tricky with the tracker.
                # Simpler: just return None and let the caller handle rollback if it took a snapshot.
                # But here we are building from scratch.
                pass
            return None
    return row

def _simulated_annealing(rows, n_rows, k, max_mark, time_budget, rnd):
    """Simulated Annealing with row reconstruction and entry shifts."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    tracker = DiffTracker(max_mark + 1)
    for r in best:
        tracker.add(_diffs_of_row(r))

    deadline = time.time() + time_budget
    T = 10.0
    cooling = 0.999
    min_T = 0.01

    # Pre-compute diffs for other rows to speed up conflict checks?
    # No, we update tracker dynamically.

    while time.time() < deadline and T > min_T:
        ri = rnd.randint(0, n_rows - 1)
        old_row = best[ri]
        old_diffs = _diffs_of_row(old_row)
        
        # Choose move type: 70% Row Rebuild, 30% Entry Shift
        if rnd.random() < 0.7:
            # Remove row from tracker
            tracker.remove(old_diffs)
            
            # Try to rebuild row with noise
            new_row = _greedy_row_with_noise(tracker, k, max_mark, rnd, noise_prob=0.3)
            
            if new_row is None:
                # Failed to rebuild, restore old row
                tracker.add(old_diffs)
                T *= cooling
                continue
            
            new_diffs = _diffs_of_row(new_row)
            new_scope = max(max(r2) for r2 in best if r2 is not old_row)
            new_scope = max(new_scope, max(new_row))
            
            delta = new_scope - best_scope
            if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                best[ri] = new_row
                tracker.add(new_diffs)
                if new_scope < best_scope:
                    best_scope = new_scope
            else:
                # Revert
                tracker.add(old_diffs)
        else:
            # Entry Shift: try to move one entry to a nearby valid position
            pi = rnd.randint(1, len(old_row) - 1)
            old_val = old_row[pi]
            
            # Try a few nearby values
            for offset in [-2, -1, 1, 2]:
                new_val = old_val + offset
                # Check bounds
                if new_val <= old_row[pi-1] or (pi < len(old_row)-1 and new_val >= old_row[pi+1]):
                    continue
                
                new_row = old_row[:]
                new_row[pi] = new_val
                new_diffs = _diffs_of_row(new_row)
                
                # Check conflicts with OTHER rows
                # We need to ensure new_diffs \ old_diffs does not conflict with (tracker - old_diffs)
                # Simpler: remove old, check new, if valid keep, else restore
                
                tracker.remove(old_diffs)
                
                # Check if new_diffs conflicts with current tracker (which now excludes old_row)
                conflict = False
                for d in new_diffs:
                    if d <= max_mark and tracker.used[d]:
                        conflict = True
                        break
                
                if not conflict:
                    new_scope = max(max(r2) for r2 in best if r2 is not old_row)
                    new_scope = max(new_scope, max(new_row))
                    delta = new_scope - best_scope
                    if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                        best[ri] = new_row
                        tracker.add(new_diffs)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        break
                    else:
                        # Revert
                        tracker.add(old_diffs)
                        break
                else:
                    # Revert
                    tracker.add(old_diffs)
                    break
            
        T *= cooling

    return best, best_scope


def _targeted_reduction(rows, n_rows, k, time_budget):
    """Systematically try to reduce each entry of each row from its current value downward."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    max_mark = best_scope

    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline:
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


def _iterative_deepening_backtrack(n_rows, k, rnd, time_budget=10.0):
    """Iterative deepening: try max_mark from lower bound upward."""
    lb = n_rows * k * (k + 1) // 2  # 105 for (7,5)
    deadline = time.time() + time_budget
    best_result = None

    for max_mark in range(lb, lb + 30):
        if time.time() > deadline:
            break
        result = _backtrack_fill(n_rows, k, max_mark, rnd, time_budget=2.0)
        if result is not None:
            best_result = result
            break

    return best_result


def construct(n_rows, k):
    rnd = random.Random(12345)
    best_rows = None
    best_scope = float('inf')

    lb = n_rows * k * (k + 1) // 2

    # Phase 1: Quick interleaved greedy to get a starting point
    rows, _ = _interleaved_greedy(n_rows, k, 200)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)

    if _dead():
        return best_rows

    # Phase 2: Targeted reduction to tighten the solution
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 2.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 3: Backtracking search at the lower bound
    remaining = _DEADLINE - time.time() - 10.0
    if remaining > 3.0:
        result = _backtrack_fill(n_rows, k, best_scope, rnd, time_budget=min(remaining, 15.0))
        if result is not None:
            sc = _scope(result)
            if sc < best_scope:
                best_scope = sc
                best_rows = result

    if _dead():
        return best_rows

    # Phase 4: Simulated Annealing for further improvement
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 3.0:
            best_rows, best_scope = _simulated_annealing(
                best_rows, n_rows, k, best_scope, remaining, rnd
            )

    if _dead():
        return best_rows

    # Phase 5: Another round of targeted reduction
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 1.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)
            
    # Phase 5.5: Final SA pass with lower temperature to tighten
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 2.0
        if remaining > 1.0:
            # Run SA again with a lower starting temperature implicitly by just calling it
            # The function resets T=10.0, so we might want a variant or just accept it.
            # Actually, let's just do one more targeted reduction if time permits.
            pass

    if _dead():
        return best_rows

    # Phase 6: Try to find solution at lower bound via more backtracking
    if best_scope > lb:
        remaining = _DEADLINE - time.time() - 2.0
        if remaining > 2.0:
            for target in range(lb, min(lb + 5, best_scope)):
                if time.time() > _DEADLINE - 2.0:
                    break
                result = _backtrack_fill(n_rows, k, target, rnd, time_budget=min(remaining / 3, 3.0))
                if result is not None:
                    sc = _scope(result)
                    if sc < best_scope:
                        best_scope = sc
                        best_rows = result
                    break

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        # Fallback: re-run greedy
        rows, _ = _interleaved_greedy(n_rows, k, 200)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
