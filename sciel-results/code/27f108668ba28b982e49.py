"""Difference triangle set (7,5) – Multi-Start Simulated Annealing:
   (a) fast interleaved greedy initialisation with adaptive bounds,
   (b) multi-start framework to escape local optima,
   (c) simulated annealing with aggressive row-rebuild and multi-row perturbation,
   (d) systematic downward compaction.
"""
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
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                return False
        return True


def _row_diffs_incremental(row, new_x):
    return {new_x - a for a in row}


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
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


def _greedy_row(tracker, k, max_mark, rnd=None, noise=0.0):
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            if noise > 0 and rnd and rnd.random() < noise:
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
    return row


def _simulated_annealing(rows, n_rows, k, max_mark, time_budget, rnd):
    """
    Simulated annealing with row-rebuild and entry-shift moves.
    Returns best rows and best scope found during this SA run.
    """
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    tracker = DiffTracker(max_mark + 1)
    for r in best:
        tracker.add(_diffs_of_row(r))

    deadline = time.time() + time_budget
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    steps_since_improve = 0
    max_steps_no_improve = 5000

    while time.time() < deadline and T > min_T:
        ri = rnd.randint(0, n_rows - 1)
        old_row = best[ri]
        old_diffs = _diffs_of_row(old_row)

        move_type = rnd.random()
        
        if move_type < 0.7:
            # Move: Rebuild a single row
            tracker.remove(old_diffs)
            noise = 0.1 + 0.4 * (T / 10.0)
            new_row = _greedy_row(tracker, k, max_mark, rnd, noise=noise)
            if new_row is None:
                tracker.add(old_diffs)
                T *= cooling
                continue
            new_diffs = _diffs_of_row(new_row)
            new_scope = max(max(r2) for r2 in best)
            new_scope = max(new_scope, max(new_row))
            delta = new_scope - best_scope
            if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                best[ri] = new_row
                tracker.add(new_diffs)
                if new_scope < best_scope:
                    best_scope = new_scope
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                tracker.add(old_diffs)
                steps_since_improve += 1
        else:
            # Move: Shift an entry
            pi = rnd.randint(1, len(old_row) - 1)
            old_val = old_row[pi]
            moved = False
            # Try small shifts first, then larger
            offsets = [-1, 1, -2, 2, -3, 3]
            rnd.shuffle(offsets)
            for offset in offsets:
                new_val = old_val + offset
                if new_val <= old_row[pi - 1]:
                    continue
                if pi < len(old_row) - 1 and new_val >= old_row[pi + 1]:
                    continue
                new_row = old_row[:]
                new_row[pi] = new_val
                new_diffs = _diffs_of_row(new_row)
                tracker.remove(old_diffs)
                conflict = False
                for d in new_diffs:
                    if d <= max_mark and tracker.used[d]:
                        conflict = True
                        break
                if not conflict:
                    new_scope = max(max(r2) for r2 in best)
                    new_scope = max(new_scope, max(new_row))
                    delta = new_scope - best_scope
                    if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                        best[ri] = new_row
                        tracker.add(new_diffs)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
                        moved = True
                        break
                    else:
                        tracker.add(old_diffs)
                        steps_since_improve += 1
                        break
                else:
                    tracker.add(old_diffs)
                    steps_since_improve += 1
                    break
            if not moved:
                T *= cooling
                steps_since_improve += 1
                
        T *= cooling
        
        if steps_since_improve > max_steps_no_improve:
            # Restart temperature if stuck
            T = 10.0
            steps_since_improve = 0

    return best, best_scope


def _targeted_reduction(rows, n_rows, k, time_budget):
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            for pi in range(len(row) - 1, 0, -1):
                old_val = row[pi]
                other_diffs = set()
                for r2 in best:
                    if r2 is not row:
                        other_diffs |= _diffs_of_row(r2)
                row_diffs_other = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if i != pi and j != pi:
                            row_diffs_other.add(row[j] - row[i])
                forbidden = other_diffs | row_diffs_other
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    new_diffs_pi = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs_pi.add(new_val - row[j])
                            else:
                                new_diffs_pi.add(row[j] - new_val)
                    if new_diffs_pi & forbidden:
                        continue
                    row[pi] = new_val
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')
    lb = n_rows * k * (k + 1) // 2

    # Initial greedy to get a baseline
    # Start with a reasonably tight bound
    initial_max_mark = 150 
    rows, _ = _interleaved_greedy(n_rows, k, initial_max_mark)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)
    else:
        # Fallback if greedy fails with 150, try higher
        rows, _ = _interleaved_greedy(n_rows, k, 200)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)
        else:
            # If even 200 fails, return empty or raise? Task says valid result.
            # Greedy with 200 should always work for 7x6.
            return None

    if _dead():
        return best_rows

    # Multi-Start Simulated Annealing
    # We will run SA for chunks of time, restarting if we haven't improved in a while
    # or just run SA until deadline, but with restarts inside.
    
    sa_time_slice = 10.0
    max_restarts = 10
    
    for restart in range(max_restarts):
        if _dead():
            break
        
        # If we have a current best, use it as the max_mark for new initializations
        # to ensure we don't get worse immediately.
        # However, for diversification, we might want to start from scratch occasionally.
        
        if restart == 0:
            current_rows = best_rows
        else:
            # Try to generate a new random solution within the current best scope
            # This forces the new solution to be at least as good as the best found so far.
            current_rows, _ = _interleaved_greedy(n_rows, k, best_scope, rnd)
            if current_rows is None:
                # If we can't find a solution within the best scope, 
                # maybe the best scope is too tight for a random greedy start.
                # In that case, just perturb the current best rows?
                # Or fall back to a looser bound.
                current_rows, _ = _interleaved_greedy(n_rows, k, int(best_scope * 1.2), rnd)
                if current_rows is None:
                    break
                # If this new solution is worse, we still use it as a starting point for SA
                # but we keep the global best separate.
            
        # Run SA
        remaining = _DEADLINE - time.time() - 5.0
        if remaining < 2.0:
            break
            
        # Limit SA time per restart to allow for multiple restarts
        sa_budget = min(sa_time_slice, remaining)
        
        # If this is the first restart, use a larger budget
        if restart == 0:
            sa_budget = min(30.0, remaining)

        sa_rows, sa_scope = _simulated_annealing(current_rows, n_rows, k, best_scope, sa_budget, rnd)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = [r[:] for r in sa_rows]
            # Optional: immediate targeted reduction after an improvement
            if not _dead() and _DEADLINE - time.time() > 5.0:
                best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, min(5.0, _DEADLINE - time.time() - 3.0))
        
        # If we hit the lower bound, we can stop early
        if best_scope == lb:
            break

    # Final polish
    if best_rows is not None and not _dead() and _DEADLINE - time.time() > 3.0:
        best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, min(3.0, _DEADLINE - time.time() - 1.0))

    if _dead():
        return best_rows

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        # Fallback to a known valid solution if something went wrong
        rows, _ = _interleaved_greedy(n_rows, k, 200)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
