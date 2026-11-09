"""Difference triangle set (7,5) – Bitset-Optimized Local Search & SA:
   (a) Fast bitset-based difference tracking for O(1) conflict checks,
   (b) Diversified interleaved greedy initialization,
   (c) Intensive local search with systematic scope reduction,
   (d) Simulated Annealing with row-rebuild and entry-shift moves.
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
        # Check differences
        for i in range(k + 1):
            ai = r[i]
            for j in range(i + 1, k + 1):
                d = r[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


class BitsetDiffTracker:
    """
    Tracks used differences using a Python integer as a bitset.
    Bit i is 1 if difference i is used.
    """
    def __init__(self, max_diff):
        self.max_diff = max_diff
        self.used = 0  # Bitset of used differences

    def add(self, diffs):
        """Adds a set of differences to the tracker."""
        mask = 0
        for d in diffs:
            if d <= self.max_diff:
                mask |= (1 << d)
        self.used |= mask

    def remove(self, diffs):
        """Removes a set of differences from the tracker."""
        mask = 0
        for d in diffs:
            if d <= self.max_diff:
                mask |= (1 << d)
        self.used &= ~mask

    def check(self, diffs):
        """Checks if a set of differences conflicts with used differences."""
        mask = 0
        for d in diffs:
            if d <= self.max_diff:
                mask |= (1 << d)
        return (self.used & mask) == 0

    def get_mask(self, diffs):
        """Returns the bitset mask for a set of differences."""
        mask = 0
        for d in diffs:
            if d <= self.max_diff:
                mask |= (1 << d)
        return mask


def _row_diffs(row):
    """Calculates the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs


def _interleaved_greedy(n_rows, k, max_mark, rnd):
    """
    Interleaved greedy construction.
    Builds all rows simultaneously, one element at a time.
    Uses bitset tracker for speed.
    """
    tracker = BitsetDiffTracker(max_mark)
    rows = [[0] for _ in range(n_rows)]
    
    # We will add k elements to each row
    for _ in range(n_rows * k):
        # Pick the row with the smallest last element to balance growth
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        
        x = row[-1] + 1
        found = False
        # To add diversity, we can sometimes skip a few valid candidates
        # but for greedy, we usually take the first one.
        while x <= max_mark:
            new_diffs = {x - a for a in row}
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


def _local_search(rows, n_rows, k, max_mark, time_budget, rnd):
    """
    Intensive local search to minimize scope.
    Systematically tries to reduce each entry in each row.
    """
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    tracker = BitsetDiffTracker(max_mark)
    for r in best:
        tracker.add(_row_diffs(r))
        
    deadline = time.time() + time_budget
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Iterate rows in random order
        row_order = list(range(n_rows))
        rnd.shuffle(row_order)
        
        for ri in row_order:
            row = best[ri]
            # Iterate positions from right to left (largest elements first)
            for pi in range(len(row) - 1, 0, -1):
                old_val = row[pi]
                # Try to reduce old_val
                # Lower bound is row[pi-1] + 1
                lower_bound = row[pi - 1] + 1
                
                # We try values from lower_bound to old_val - 1
                # To be efficient, we can just check if old_val - 1 is valid, 
                # but often we need to jump further.
                # Let's try a few candidates.
                
                # Calculate diffs for current position
                old_diffs = set()
                for j in range(len(row)):
                    if j != pi:
                        if j < pi:
                            old_diffs.add(old_val - row[j])
                        else:
                            old_diffs.add(row[j] - old_val)
                
                # Temporarily remove old diffs
                tracker.remove(old_diffs)
                
                found_new_val = None
                # Try to find the smallest valid value >= lower_bound
                # We scan upwards from lower_bound.
                # To speed up, we might check old_val - 1, old_val - 2, etc.
                # But scanning from lower_bound is safer for global minimization.
                # However, scanning from lower_bound every time is O(N^2).
                # Let's try a hybrid: check old_val-1, then if fails, scan from lower_bound.
                
                # Optimization: Only scan if we are close to the bound or if we want to be thorough.
                # Given the small size (6 elements), scanning is fast enough.
                for new_val in range(lower_bound, old_val):
                    new_diffs = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs.add(new_val - row[j])
                            else:
                                new_diffs.add(row[j] - new_val)
                    
                    if tracker.check(new_diffs):
                        found_new_val = new_val
                        break
                
                if found_new_val is not None:
                    # Apply move
                    row[pi] = found_new_val
                    new_diffs = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs.add(found_new_val - row[j])
                            else:
                                new_diffs.add(row[j] - found_new_val)
                    tracker.add(new_diffs)
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                else:
                    # Restore old diffs
                    tracker.add(old_diffs)
                    # No improvement for this position
                    # We don't break the position loop immediately because 
                    # reducing a later element might allow reducing an earlier one? 
                    # No, we are going right-to-left. Reducing a right element 
                    # doesn't affect the validity of left elements' reductions 
                    # in terms of bounds, but it frees up differences.
                    # However, if we didn't find a reduction for pi, we move to pi-1.
                    
    return best, best_scope


def _simulated_annealing(rows, n_rows, k, max_mark, time_budget, rnd):
    """
    Simulated Annealing with row-rebuild and entry-shift moves.
    Uses bitset tracker for speed.
    """
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    tracker = BitsetDiffTracker(max_mark)
    for r in best:
        tracker.add(_row_diffs(r))

    deadline = time.time() + time_budget
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    steps_since_improve = 0
    max_steps_no_improve = 5000

    while time.time() < deadline and T > min_T:
        ri = rnd.randint(0, n_rows - 1)
        old_row = best[ri]
        old_diffs = _row_diffs(old_row)

        move_type = rnd.random()
        
        if move_type < 0.7:
            # Move: Rebuild a single row
            tracker.remove(old_diffs)
            # Use a greedy-like approach but with some randomness to diversify
            noise = 0.1 + 0.4 * (T / 10.0)
            new_row = [0]
            valid = True
            for _ in range(k):
                x = new_row[-1] + 1
                found = False
                while x <= max_mark:
                    # Skip some candidates based on noise
                    if rnd.random() < noise:
                        x += 1
                        continue
                    new_diffs = {x - a for a in new_row}
                    if tracker.check(new_diffs):
                        new_row.append(x)
                        tracker.add(new_diffs)
                        found = True
                        break
                    x += 1
                if not found:
                    valid = False
                    break
            
            if valid:
                new_diffs = _row_diffs(new_row)
                new_scope = max(max(r2) for r2 in best)
                new_scope = max(new_scope, max(new_row))
                delta = new_scope - best_scope
                if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                    best[ri] = new_row
                    # Note: tracker already has new_row diffs added during construction
                    # We removed old_row diffs before. So state is consistent.
                    if new_scope < best_scope:
                        best_scope = new_scope
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    # Reject move: Revert tracker to old state
                    tracker.remove(new_diffs)
                    tracker.add(old_diffs)
                    steps_since_improve += 1
            else:
                # Invalid new row: Restore old row
                tracker.add(old_diffs)
                T *= cooling
                steps_since_improve += 1
        else:
            # Move: Shift an entry
            pi = rnd.randint(1, len(old_row) - 1)
            old_val = old_row[pi]
            moved = False
            
            # Try a random shift
            shift_range = int(5 + 10 * (T / 10.0))
            offset = rnd.randint(-shift_range, shift_range)
            new_val = old_val + offset
            
            if new_val > old_row[pi - 1] and (pi == len(old_row) - 1 or new_val < old_row[pi + 1]):
                new_row = old_row[:]
                new_row[pi] = new_val
                new_diffs = _row_diffs(new_row)
                
                tracker.remove(old_diffs)
                if tracker.check(new_diffs):
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
                    else:
                        # Reject
                        # old_diffs are already removed, new_diffs not added
                        # We just need to re-add old_diffs
                        tracker.add(old_diffs)
                        steps_since_improve += 1
                else:
                    # Conflict
                    tracker.add(old_diffs)
                    steps_since_improve += 1
            else:
                T *= cooling
                steps_since_improve += 1
                
        T *= cooling
        
        if steps_since_improve > max_steps_no_improve:
            # Restart temperature if stuck
            T = 10.0
            steps_since_improve = 0

    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')
    
    # Initial greedy to get a baseline
    # Start with a reasonably tight bound
    initial_max_mark = 150 
    rows, _ = _interleaved_greedy(n_rows, k, initial_max_mark, rnd)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)
    else:
        # Fallback if greedy fails with 150, try higher
        rows, _ = _interleaved_greedy(n_rows, k, 200, rnd)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)
        else:
            return None

    if _dead():
        return best_rows

    # Multi-Start Optimization
    # We alternate between Local Search (intensive reduction) and SA (exploration)
    
    sa_time_slice = 10.0
    ls_time_slice = 5.0
    max_restarts = 10
    
    for restart in range(max_restarts):
        if _dead():
            break
        
        if restart == 0:
            current_rows = best_rows
        else:
            # Generate a new random solution within the current best scope
            # This forces the new solution to be at least as good as the best found so far.
            current_rows, _ = _interleaved_greedy(n_rows, k, best_scope, rnd)
            if current_rows is None:
                # If we can't find a solution within the best scope, 
                # maybe the best scope is too tight for a random greedy start.
                current_rows, _ = _interleaved_greedy(n_rows, k, int(best_scope * 1.2), rnd)
                if current_rows is None:
                    break
            
        # Run Local Search first to polish the initial solution
        remaining = _DEADLINE - time.time() - 5.0
        if remaining < 2.0:
            break
            
        ls_budget = min(ls_time_slice, remaining / 2.0)
        ls_rows, ls_scope = _local_search(current_rows, n_rows, k, best_scope, ls_budget, rnd)
        
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [r[:] for r in ls_rows]
        
        # Run SA to explore further
        remaining = _DEADLINE - time.time() - 5.0
        if remaining < 2.0:
            break
            
        sa_budget = min(sa_time_slice, remaining / 2.0)
        sa_rows, sa_scope = _simulated_annealing(best_rows, n_rows, k, best_scope, sa_budget, rnd)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = [r[:] for r in sa_rows]
            # Optional: immediate local search after SA improvement
            if not _dead() and _DEADLINE - time.time() > 5.0:
                ls_rows, ls_scope = _local_search(best_rows, n_rows, k, best_scope, min(5.0, _DEADLINE - time.time() - 3.0), rnd)
                if ls_scope < best_scope:
                    best_scope = ls_scope
                    best_rows = [r[:] for r in ls_rows]
        
        # If we hit the lower bound, we can stop early
        lb = n_rows * k * (k + 1) // 2
        if best_scope == lb:
            break

    # Final polish
    if best_rows is not None and not _dead() and _DEADLINE - time.time() > 3.0:
        best_rows, best_scope = _local_search(best_rows, n_rows, k, best_scope, min(3.0, _DEADLINE - time.time() - 1.0), rnd)

    if _dead():
        return best_rows

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        # Fallback to a known valid solution if something went wrong
        rows, _ = _interleaved_greedy(n_rows, k, 200, rnd)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
