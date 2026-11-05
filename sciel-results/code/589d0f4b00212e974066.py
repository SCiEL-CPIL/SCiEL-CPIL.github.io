<<<<<<< SEARCH
"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
from collections import defaultdict

_TIME_LIMIT = 150.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _row_diffs(row):
    """Compute set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs_except(rows, skip_idx):
    """Compute set of all differences except those in rows[skip_idx]."""
    s = set()
    for idx, row in enumerate(rows):
        if idx == skip_idx:
            continue
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
    return s


def _greedy_row(used, k, rng=None, strategy="pure"):
    """
    Build one row greedily.
    strategy: 'pure' (smallest valid), 'shuffled' (random from window),
              'lookahead' (score candidates by future options)
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        elif strategy == "shuffled":
            base = x + 1
            window = 30
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= {cand - a for a in row[:-1]}
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 40):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 6:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append(cand)
                        break
                    cand += 1
            
            best_cand = valid_cands[0]
            best_score = -1
            for cand in valid_cands[:4]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 20):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 3:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
    return row


def _backtrack_row(used, k, max_scope, rng):
    """
    Try to build one row with all entries <= max_scope using backtracking.
    Returns row or None.
    """
    row = [0]
    
    def bt():
        if len(row) == k + 1:
            return True
        start = row[-1] + 1
        end = min(max_scope + 1, start + 50)
        candidates = list(range(start, end))
        if rng and len(candidates) > 8:
            rng.shuffle(candidates[4:])
        for x in candidates:
            if x > max_scope:
                break
            new = {x - a for a in row}
            if new & used:
                continue
            row.append(x)
            used.update(new)
            if bt():
                return True
            row.pop()
            used.difference_update(new)
        return False
    
    if bt():
        return row[:]
    return None


def _construct_diverse(n_rows, k, rng, strategy):
    """Construct full set using a given strategy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, strategy)
        rows.append(row)
    return rows


def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """
    Simulated Annealing to minimize the scope.
    State: list of n_rows rows.
    Energy: scope (max entry).
    Moves:
      1. Rebuild one row via randomized backtracking within current scope
      2. Rebuild one row with a slightly larger scope (to escape)
      3. Perturb a single element of a row
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    
    # Temperature schedule
    T0 = max(10.0, current_scope * 0.1)
    T_end = 0.1
    total_iters = 50000
    start_time = time.time()
    
    for it in range(total_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Cooling schedule (geometric)
        progress = it / total_iters
        T = T0 * (T_end / T0) ** progress
        
        # Choose a move
        move_type = rng.random()
        row_idx = rng.randint(0, n_rows - 1)
        
        new_rows = [r[:] for r in current_rows]
        new_scope = current_scope
        accepted = False
        
        if move_type < 0.5:
            # Move 1: Rebuild row with current best scope
            used_other = _all_diffs_except(current_rows, row_idx)
            new_row = _backtrack_row(used_other, k, best_scope, rng)
            if new_row is not None:
                new_rows[row_idx] = new_row
                new_scope = _scope(new_rows)
                accepted = True
        elif move_type < 0.75:
            # Move 2: Rebuild row with slightly larger scope (escape)
            escape_scope = min(best_scope + rng.randint(1, 10), current_scope + 5)
            used_other = _all_diffs_except(current_rows, row_idx)
            new_row = _backtrack_row(used_other, k, escape_scope, rng)
            if new_row is not None:
                new_rows[row_idx] = new_row
                new_scope = _scope(new_rows)
                accepted = True
        else:
            # Move 3: Perturb a single element of a row
            row = current_rows[row_idx]
            if len(row) > 1:
                pos = rng.randint(1, len(row) - 1)  # Don't change first element (0)
                old_val = row[pos]
                # Try a random new value
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 5
                if upper > lower:
                    new_val = rng.randint(lower, upper)
                    if new_val != old_val:
                        new_rows[row_idx] = row[:]
                        new_rows[row_idx][pos] = new_val
                        # Validate this row
                        internal = _row_diffs(new_rows[row_idx])
                        expected = len(row) * (len(row) - 1) // 2
                        if len(internal) == expected:
                            used_other = _all_diffs_except(current_rows, row_idx)
                            if not (internal & used_other):
                                new_scope = _scope(new_rows)
                                accepted = True
        
        if not accepted:
            continue
        
        # Accept/reject based on energy
        delta = new_scope - current_scope
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current_rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
    
    return best_rows, best_scope


def _local_search_refine(rows, n_rows, k, rng, max_iters=20000):
    """
    Local search refinement: try to reduce the largest entry by 
    rebuilding rows with tighter scope bounds.
    """
    best_scope = _scope(rows)
    rows = [r[:] for r in rows]
    iters = 0
    
    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        
        # Find the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        if current_max <= best_scope:
            target = best_scope - 1
        else:
            target = best_scope - 1
        
        if target < k + 1:
            break
        
        # Compute differences used by other rows
        used_other = _all_diffs_except(rows, max_idx)
        
        # Try to rebuild the max row with smaller scope
        new_row = _backtrack_row(used_other, k, target, rng)
        if new_row is not None:
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            continue
        
        # If rebuild fails, try reducing individual elements
        row = rows[max_idx]
        improved = False
        
        # Try reducing the last element
        for new_last in range(current_max - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            new_row = row[:-1] + [new_last]
            internal = _row_diffs(new_row)
            expected = len(new_row) * (len(new_row) - 1) // 2
            if len(internal) != expected:
                continue
            if internal & used_other:
                continue
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break
        
        if not improved:
            # Try reducing other elements
            found_move = False
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound:
                        break
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    internal = _row_diffs(new_row)
                    expected = len(new_row) * (len(new_row) - 1) // 2
                    if len(internal) != expected:
                        continue
                    if internal & used_other:
                        continue
                    rows[max_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    found_move = True
                    break
                if found_move:
                    break
            
            if not found_move:
                break
    
    return rows, best_scope


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)
    
    # Phase 1: Multiple diverse constructions
    best_rows = None
    best_scope = float('inf')
    
    strategies = ["pure", "shuffled", "lookahead"]
    for strategy in strategies:
        if _time_left() < 30.0:
            break
        rows = _construct_diverse(n_rows, k, rng, strategy)
        if _validate(rows):
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Fallback if all failed
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, "pure")
            best_rows.append(row)
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated Annealing (structural change)
    if _time_left() > 40.0:
        sa_budget = min(_time_left() - 10.0, 60.0)
        rows_sa, scope_sa = _simulated_annealing(best_rows, n_rows, k, rng, sa_budget)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa
    
    # Phase 3: Local search refinement
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search_refine(best_rows, n_rows, k, rng)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls
    
    # Phase 4: Another SA pass if time allows
    if _time_left() > 25.0:
        sa_budget = min(_time_left() - 10.0, 30.0)
        rows_sa2, scope_sa2 = _simulated_annealing(best_rows, n_rows, k, rng, sa_budget)
        if scope_sa2 < best_scope:
            best_scope = scope_sa2
            best_rows = rows_sa2
    
    # Final validation
    if not _validate(best_rows):
        # Return a valid fallback
        used = set()
        fallback_rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, "pure")
            fallback_rows.append(row)
        return fallback_rows
    
    return best_rows
=======
"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
import sys

_TIME_LIMIT = 150.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _row_diffs(row):
    """Compute set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs_except(rows, skip_idx):
    """Compute set of all differences except those in rows[skip_idx]."""
    s = set()
    for idx, row in enumerate(rows):
        if idx == skip_idx:
            continue
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
    return s


class _BitsetSolver:
    """
    CSP solver using bitsets for fast difference conflict checking.
    Uses forward checking with dynamic variable ordering.
    """
    
    def __init__(self, n_rows, k, max_scope, rng):
        self.n_rows = n_rows
        self.k = k
        self.max_scope = max_scope
        self.rng = rng
        # Bitset of used differences: bit d set means difference d is used
        # We need differences up to max_scope
        self.used_bits = 0
        self.rows = [[0] for _ in range(n_rows)]
        self.row_len = [1] * n_rows
        self._nodes = 0
        self._max_nodes = 5000000  # Node limit to prevent timeout
    
    def _diff_conflicts(self, row, pos, candidate):
        """Check if placing candidate at position pos in row conflicts with used differences."""
        # Compute new differences: candidate - row[i] for i < pos
        for i in range(pos):
            d = candidate - row[i]
            if d <= 0 or d > self.max_scope:
                return False
            if (self.used_bits >> d) & 1:
                return False
        return True
    
    def _place(self, row, pos, candidate):
        """Place candidate and update used_bits."""
        for i in range(pos):
            d = candidate - row[i]
            self.used_bits |= (1 << d)
        row.append(candidate)
        self.row_len[row.__class__.__mro__[0].__name__ and 0]  # placeholder
    
    def _unplace(self, row, pos, candidate):
        """Remove candidate and update used_bits."""
        for i in range(pos):
            d = candidate - row[i]
            self.used_bits &= ~(1 << d)
        row.pop()
    
    def _forward_check(self, row_idx, pos, candidate):
        """
        Forward check: after placing candidate, verify that for each remaining
        position in this row and for other rows, there exists at least one valid candidate.
        Returns True if feasible, False if dead end.
        """
        # Check remaining positions in current row
        for future_pos in range(pos + 1, self.k + 1):
            lower = candidate + 1
            upper = self.max_scope
            found = False
            for c in range(lower, upper + 1):
                if self._diff_conflicts(row, future_pos, c):
                    found = True
                    break
            if not found:
                return False
        
        # Check other rows (lighter check: just verify first unfilled position has options)
        for other_idx in range(self.n_rows):
            if other_idx == row_idx:
                continue
            other_row = self.rows[other_idx]
            other_pos = len(other_row)
            if other_pos >= self.k + 1:
                continue
            lower = other_row[-1] + 1
            upper = self.max_scope
            found = False
            for c in range(lower, upper + 1):
                if self._diff_conflicts(other_row, other_pos, c):
                    found = True
                    break
            if not found:
                return False
        
        return True
    
    def _select_next(self):
        """
        Dynamic variable ordering: select the (row, position) with fewest valid candidates.
        Returns (row_idx, pos) or None if all rows are complete.
        """
        best = None
        best_count = float('inf')
        
        for r in range(self.n_rows):
            pos = len(self.rows[r])
            if pos >= self.k + 1:
                continue
            
            # Count valid candidates (with early termination)
            lower = self.rows[r][-1] + 1
            upper = self.max_scope
            count = 0
            for c in range(lower, upper + 1):
                if self._diff_conflicts(self.rows[r], pos, c):
                    count += 1
                    if count >= best_count:
                        break
            
            if count == 0:
                return (r, pos)  # Dead end
            
            if count < best_count:
                best_count = count
                best = (r, pos)
        
        return best
    
    def solve(self):
        """
        Depth-first search with forward checking and dynamic variable ordering.
        Returns list of rows if solution found, else None.
        """
        self._nodes = 0
        
        def dfs():
            self._nodes += 1
            if self._nodes > self._max_nodes:
                return False
            
            # Check if all rows are complete
            for r in range(self.n_rows):
                if len(self.rows[r]) < self.k + 1:
                    break
            else:
                return True
            
            # Select next variable
            var = self._select_next()
            if var is None:
                return True
            
            row_idx, pos = var
            row = self.rows[row_idx]
            lower = row[-1] + 1
            upper = self.max_scope
            
            # Collect valid candidates
            candidates = []
            for c in range(lower, upper + 1):
                if self._diff_conflicts(row, pos, c):
                    candidates.append(c)
            
            if not candidates:
                return False
            
            # Order candidates: prefer smaller values (to minimize scope)
            # But with some randomization for diversity
            if self.rng and len(candidates) > 5:
                # Keep first few in order, shuffle the rest
                split = min(5, len(candidates))
                self.rng.shuffle(candidates[split:])
            
            for c in candidates:
                # Place
                self._place(row, pos, c)
                
                # Forward check
                if self._forward_check(row_idx, pos, c):
                    if dfs():
                        return True
                
                # Unplace
                self._unplace(row, pos, c)
            
            return False
        
        if dfs():
            return [r[:] for r in self.rows]
        return None


def _greedy_row(used, k, rng=None, strategy="pure"):
    """
    Build one row greedily.
    strategy: 'pure' (smallest valid), 'shuffled' (random from window),
              'lookahead' (score candidates by future options)
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        elif strategy == "shuffled":
            base = x + 1
            window = 30
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= {cand - a for a in row[:-1]}
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 40):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 6:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append(cand)
                        break
                    cand += 1
            
            best_cand = valid_cands[0]
            best_score = -1
            for cand in valid_cands[:4]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 20):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 3:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
    return row


def _backtrack_row(used, k, max_scope, rng):
    """
    Try to build one row with all entries <= max_scope using backtracking.
    Returns row or None.
    """
    row = [0]
    
    def bt():
        if len(row) == k + 1:
            return True
        start = row[-1] + 1
        end = min(max_scope + 1, start + 50)
        candidates = list(range(start, end))
        if rng and len(candidates) > 8:
            rng.shuffle(candidates[4:])
        for x in candidates:
            if x > max_scope:
                break
            new = {x - a for a in row}
            if new & used:
                continue
            row.append(x)
            used.update(new)
            if bt():
                return True
            row.pop()
            used.difference_update(new)
        return False
    
    if bt():
        return row[:]
    return None


def _construct_diverse(n_rows, k, rng, strategy):
    """Construct full set using a given strategy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, strategy)
        rows.append(row)
    return rows


def _csp_solve(n_rows, k, max_scope, rng, node_limit=5000000):
    """
    Use the bitset CSP solver to find a solution with scope <= max_scope.
    Returns list of rows or None.
    """
    solver = _BitsetSolver(n_rows, k, max_scope, rng)
    solver._max_nodes = node_limit
    return solver.solve()


def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """
    Simulated Annealing to minimize the scope.
    State: list of n_rows rows.
    Energy: scope (max entry).
    Moves:
      1. Rebuild one row via CSP solver within current scope
      2. Rebuild one row with a slightly larger scope (to escape)
      3. Perturb a single element of a row
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    
    # Temperature schedule
    T0 = max(10.0, current_scope * 0.1)
    T_end = 0.1
    total_iters = 50000
    start_time = time.time()
    
    for it in range(total_iters):
        if time.time() - start_time > time_budget:
            break
        if _time_left() < 5.0:
            break
        
        # Cooling schedule (geometric)
        progress = it / total_iters
        T = T0 * (T_end / T0) ** progress
        
        # Choose a move
        move_type = rng.random()
        row_idx = rng.randint(0, n_rows - 1)
        
        new_rows = [r[:] for r in current_rows]
        new_scope = current_scope
        accepted = False
        
        if move_type < 0.6:
            # Move 1: Rebuild row using CSP solver with current best scope
            used_other = _all_diffs_except(current_rows, row_idx)
            
            # Use CSP solver to rebuild this row
            # We need to construct a partial problem: fix other rows, solve for this row
            # For simplicity, use backtracking for single row
            new_row = _backtrack_row(used_other, k, best_scope, rng)
            if new_row is not None:
                new_rows[row_idx] = new_row
                new_scope = _scope(new_rows)
                accepted = True
        elif move_type < 0.8:
            # Move 2: Rebuild row with slightly larger scope (escape)
            escape_scope = min(best_scope + rng.randint(1, 10), current_scope + 5)
            used_other = _all_diffs_except(current_rows, row_idx)
            new_row = _backtrack_row(used_other, k, escape_scope, rng)
            if new_row is not None:
                new_rows[row_idx] = new_row
                new_scope = _scope(new_rows)
                accepted = True
        else:
            # Move 3: Perturb a single element of a row
            row = current_rows[row_idx]
            if len(row) > 1:
                pos = rng.randint(1, len(row) - 1)  # Don't change first element (0)
                old_val = row[pos]
                # Try a random new value
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 5
                if upper > lower:
                    new_val = rng.randint(lower, upper)
                    if new_val != old_val:
                        new_rows[row_idx] = row[:]
                        new_rows[row_idx][pos] = new_val
                        # Validate this row
                        internal = _row_diffs(new_rows[row_idx])
                        expected = len(row) * (len(row) - 1) // 2
                        if len(internal) == expected:
                            used_other = _all_diffs_except(current_rows, row_idx)
                            if not (internal & used_other):
                                new_scope = _scope(new_rows)
                                accepted = True
        
        if not accepted:
            continue
        
        # Accept/reject based on energy
        delta = new_scope - current_scope
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current_rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
    
    return best_rows, best_scope


def _local_search_refine(rows, n_rows, k, rng, max_iters=20000):
    """
    Local search refinement: try to reduce the largest entry by 
    rebuilding rows with tighter scope bounds.
    """
    best_scope = _scope(rows)
    rows = [r[:] for r in rows]
    iters = 0
    
    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        
        # Find the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        target = best_scope - 1
        
        if target < k + 1:
            break
        
        # Compute differences used by other rows
        used_other = _all_diffs_except(rows, max_idx)
        
        # Try to rebuild the max row with smaller scope using backtracking
        new_row = _backtrack_row(used_other, k, target, rng)
        if new_row is not None:
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            continue
        
        # If rebuild fails, try reducing individual elements
        row = rows[max_idx]
        improved = False
        
        # Try reducing the last element
        for new_last in range(current_max - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            new_row = row[:-1] + [new_last]
            internal = _row_diffs(new_row)
            expected = len(new_row) * (len(new_row) - 1) // 2
            if len(internal) != expected:
                continue
            if internal & used_other:
                continue
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break
        
        if not improved:
            # Try reducing other elements
            found_move = False
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound:
                        break
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    internal = _row_diffs(new_row)
                    expected = len(new_row) * (len(new_row) - 1) // 2
                    if len(internal) != expected:
                        continue
                    if internal & used_other:
                        continue
                    rows[max_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    found_move = True
                    break
                if found_move:
                    break
            
            if not found_move:
                break
    
    return rows, best_scope


def _csp_based_construction(n_rows, k, rng, initial_scope=None):
    """
    Use CSP solver to construct the full set.
    Tries progressively larger scopes until a solution is found.
    """
    if initial_scope is None:
        # Estimate initial scope from counting bound
        # Total differences needed: n_rows * k*(k+1)/2
        # Each difference d can be used at most once
        # Need n_rows * k*(k+1)/2 distinct differences from 1..m
        # So m >= n_rows * k*(k+1)/2
        total_diffs = n_rows * k * (k + 1) // 2
        initial_scope = total_diffs + k  # Add margin
    
    # Try increasing scopes
    scope = initial_scope
    max_attempts = 20
    
    for attempt in range(max_attempts):
        if _time_left() < 10.0:
            break
        
        result = _csp_solve(n_rows, k, scope, rng, node_limit=2000000)
        if result is not None:
            if _validate(result):
                return result, scope
        
        # Increase scope
        scope += max(5, scope // 10)
    
    return None, None


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)
    
    # Phase 1: Try CSP-based construction first
    best_rows = None
    best_scope = float('inf')
    
    if _time_left() > 30.0:
        csp_rows, csp_scope = _csp_based_construction(n_rows, k, rng)
        if csp_rows is not None:
            best_rows = csp_rows
            best_scope = csp_scope
    
    # Phase 2: If CSP didn't work well, try greedy constructions
    if best_rows is None or best_scope > 150:
        strategies = ["pure", "shuffled", "lookahead"]
        for strategy in strategies:
            if _time_left() < 20.0:
                break
            rows = _construct_diverse(n_rows, k, rng, strategy)
            if _validate(rows):
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
    
    # Fallback if all failed
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, "pure")
            best_rows.append(row)
        best_scope = _scope(best_rows)
    
    # Phase 3: Simulated Annealing (structural change)
    if _time_left() > 30.0:
        sa_budget = min(_time_left() - 10.0, 50.0)
        rows_sa, scope_sa = _simulated_annealing(best_rows, n_rows, k, rng, sa_budget)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa
    
    # Phase 4: Local search refinement
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search_refine(best_rows, n_rows, k, rng)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls
    
    # Phase 5: Another SA pass if time allows
    if _time_left() > 20.0:
        sa_budget = min(_time_left() - 10.0, 25.0)
        rows_sa2, scope_sa2 = _simulated_annealing(best_rows, n_rows, k, rng, sa_budget)
        if scope_sa2 < best_scope:
            best_scope = scope_sa2
            best_rows = rows_sa2
    
    # Final validation
    if not _validate(best_rows):
        # Return a valid fallback
        used = set()
        fallback_rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, "pure")
            fallback_rows.append(row)
        return fallback_rows
    
    return best_rows
>>>>>>> REPLACE
