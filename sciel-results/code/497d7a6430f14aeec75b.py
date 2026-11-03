import time
import random
import math
from collections import Counter

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng=None, spread=False):
    """Bitmask-based greedy construction for fast initialization."""
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

class _SAState:
    """Fast incremental state tracking for simulated annealing."""
    def __init__(self, rows, k):
        self.rows = [list(r) for r in rows]
        self.k = k
        self.row_len = k + 1
        self.n_rows = len(rows)
        self.diff_count = {}
        self.collisions = 0
        self.max_val = 0
        self._rebuild()
    
    def _rebuild(self):
        self.diff_count = {}
        self.collisions = 0
        self.max_val = 0
        for r in self.rows:
            if r[-1] > self.max_val:
                self.max_val = r[-1]
            for i in range(self.row_len):
                for j in range(i + 1, self.row_len):
                    d = r[j] - r[i]
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1
    
    def get_cost(self):
        return self.collisions, self.max_val
    
    def is_valid(self):
        return self.collisions == 0
    
    def _remove_element_diffs(self, r_idx, i_idx, val):
        """Remove all differences involving rows[r_idx][i_idx] = val."""
        row = self.rows[r_idx]
        for j in range(self.row_len):
            if j != i_idx:
                d = abs(val - row[j])
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
    
    def _add_element_diffs(self, r_idx, i_idx, val):
        """Add all differences involving rows[r_idx][i_idx] = val."""
        row = self.rows[r_idx]
        for j in range(self.row_len):
            if j != i_idx:
                d = abs(val - row[j])
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
    
    def try_move(self, r_idx, i_idx, new_val):
        """Try to change rows[r_idx][i_idx] to new_val. Returns (old_val, success)."""
        old_val = self.rows[r_idx][i_idx]
        if old_val == new_val:
            return old_val, False
        
        # Check bounds
        if i_idx > 0 and new_val <= self.rows[r_idx][i_idx - 1]:
            return old_val, False
        if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx + 1]:
            return old_val, False
        
        # Remove old differences
        self._remove_element_diffs(r_idx, i_idx, old_val)
        
        # Apply change
        old_max = self.max_val
        self.rows[r_idx][i_idx] = new_val
        
        # Add new differences
        self._add_element_diffs(r_idx, i_idx, new_val)
        
        # Update max_val
        if new_val > self.max_val:
            self.max_val = new_val
        elif old_val == old_max and new_val < old_max:
            self.max_val = max(r[-1] for r in self.rows)
        
        return old_val, True
    
    def undo_move(self, r_idx, i_idx, old_val):
        """Undo a move by restoring old_val."""
        new_val = self.rows[r_idx][i_idx]
        if new_val == old_val:
            return
        
        # Remove new differences
        self._remove_element_diffs(r_idx, i_idx, new_val)
        
        # Restore old value
        self.rows[r_idx][i_idx] = old_val
        
        # Add old differences
        self._add_element_diffs(r_idx, i_idx, old_val)
        
        # Update max_val
        if old_val > self.max_val:
            self.max_val = old_val
        elif new_val == self.max_val and old_val < new_val:
            self.max_val = max(r[-1] for r in self.rows)
    
    def rebuild_row(self, r_idx, new_row):
        """Replace an entire row. Returns old row."""
        old_row = list(self.rows[r_idx])
        
        # Remove all differences from old row
        for i in range(self.row_len):
            for j in range(i + 1, self.row_len):
                d = old_row[j] - old_row[i]
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
        
        # Replace row
        self.rows[r_idx] = new_row
        
        # Add all differences from new row
        for i in range(self.row_len):
            for j in range(i + 1, self.row_len):
                d = new_row[j] - new_row[i]
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
        
        # Update max_val
        if new_row[-1] > self.max_val:
            self.max_val = new_row[-1]
        elif old_row[-1] == self.max_val and new_row[-1] < old_row[-1]:
            self.max_val = max(r[-1] for r in self.rows)
        
        return old_row

def _csp_build_row(k, used_diffs, max_val, rng, time_limit):
    """
    Build a single row [0, a1, ..., ak] using CSP backtracking.
    used_diffs: set of already-used differences (globally).
    Returns the row or None if not found within time_limit seconds.
    """
    start = time.time()
    best_row = None
    best_score = float('inf')
    
    for attempt in range(30):
        if time.time() - start > time_limit:
            break
        
        row = [0]
        local_used = set()
        ok = True
        
        for j in range(1, k + 1):
            found = False
            min_val = row[-1] + 1
            max_val_candidate = max_val - (k - j)
            
            if max_val_candidate < min_val:
                ok = False
                break
            
            if attempt < 5:
                candidates = list(range(min_val, max_val_candidate + 1))
            else:
                candidates = list(range(min_val, max_val_candidate + 1))
                rng.shuffle(candidates)
            
            for v in candidates:
                valid = True
                new_diffs = []
                for a in row:
                    d = v - a
                    if d in used_diffs or d in local_used:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(v)
                    local_used.update(new_diffs)
                    found = True
                    break
            
            if not found:
                ok = False
                break
        
        if ok:
            score = row[-1]
            if score < best_score:
                best_score = score
                best_row = list(row)
                if best_score <= k * (k + 1) // 2:
                    break
    
    return best_row

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    """Simulated annealing with fast incremental state updates."""
    state = _SAState(rows, k)
    current_collisions, current_max = state.get_cost()
    best_collisions = current_collisions
    best_max = current_max
    best_rows = [list(r) for r in state.rows]
    
    if polish:
        T = 2.0
        cooling = 0.99995
        min_T = 0.05
    else:
        T = 20.0
        cooling = 0.9998
        min_T = 0.1
    
    n_rows = state.n_rows
    row_len = state.row_len
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        r_idx = rng.randint(0, n_rows - 1)
        i_idx = rng.randint(1, row_len - 1)
        
        lower = state.rows[r_idx][i_idx - 1] + 1
        if i_idx < row_len - 1:
            upper = state.rows[r_idx][i_idx + 1] - 1
        else:
            upper = best_max + 5
        
        if lower > upper:
            continue
        
        # Try multiple candidate values
        for _ in range(4):
            new_val = rng.randint(lower, upper)
            old_val, success = state.try_move(r_idx, i_idx, new_val)
            if not success:
                continue
            
            new_collisions, new_max = state.get_cost()
            
            # Cost: prioritize eliminating collisions, then minimize max
            old_cost = current_collisions * 1000 + current_max
            new_cost = new_collisions * 1000 + new_max
            
            if new_cost < old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
                current_collisions, current_max = new_collisions, new_max
                if new_cost < best_collisions * 1000 + best_max:
                    best_collisions = new_collisions
                    best_max = new_max
                    best_rows = [list(r) for r in state.rows]
            else:
                state.undo_move(r_idx, i_idx, old_val)
            break  # Only one move per iteration
        
        T = max(T * cooling, min_T)
    
    return best_max, best_rows

def _row_rebuild_sa(rows, k, rng, deadline, max_scope):
    """Iteratively rebuild rows using CSP to reduce scope."""
    state = _SAState(rows, k)
    best_rows = [list(r) for r in state.rows]
    best_scope = state.max_val
    
    n_rows = state.n_rows
    
    while time.time() < deadline:
        improved = False
        for r_idx in range(n_rows):
            if time.time() > deadline:
                break
            
            old_row = list(state.rows[r_idx])
            used_diffs = set(state.diff_count.keys())
            
            # Try to rebuild with tighter scope
            new_row = _csp_build_row(k, used_diffs, max_scope, rng, 0.08)
            if new_row is None:
                new_row = _csp_build_row(k, used_diffs, max_scope + 5, rng, 0.08)
            
            if new_row is not None:
                old_row_saved = state.rebuild_row(r_idx, new_row)
                _, new_max = state.get_cost()
                
                if new_max < best_scope:
                    best_scope = new_max
                    best_rows = [list(r) for r in state.rows]
                    improved = True
                else:
                    # Revert if not better
                    state.rebuild_row(r_idx, old_row_saved)
        
        if not improved:
            break
    
    return best_scope, best_rows

def _final_polish(rows, k, deadline, rng):
    """Aggressive local search to compress scope further."""
    state = _SAState(rows, k)
    best_rows = [list(r) for r in state.rows]
    best_scope = state.max_val
    
    n_rows = state.n_rows
    row_len = state.row_len
    T = 1.0
    cooling = 0.99998
    min_T = 0.02
    
    while time.time() < deadline:
        r_idx = rng.randint(0, n_rows - 1)
        i_idx = rng.randint(1, row_len - 1)
        
        lower = state.rows[r_idx][i_idx - 1] + 1
        if i_idx < row_len - 1:
            upper = state.rows[r_idx][i_idx + 1] - 1
        else:
            upper = best_scope
        
        if lower > upper:
            continue
        
        # Prefer decreasing values for compression
        if state.rows[r_idx][i_idx] > lower:
            new_val = rng.randint(lower, state.rows[r_idx][i_idx] - 1)
        else:
            new_val = rng.randint(state.rows[r_idx][i_idx] + 1, upper)
        
        old_val, success = state.try_move(r_idx, i_idx, new_val)
        if not success:
            continue
        
        new_collisions, new_max = state.get_cost()
        old_cost = state.collisions * 1000 + state.max_val
        new_cost = new_collisions * 1000 + new_max
        
        if new_cost < old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
            if new_cost < best_collisions * 1000 + best_scope if hasattr(state, 'collisions') else True:
                best_collisions = new_collisions
                best_scope = new_max
                best_rows = [list(r) for r in state.rows]
        else:
            state.undo_move(r_idx, i_idx, old_val)
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 117.0  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions via fast greedy
    init_deadline = start_time + 3.0
    candidates = []
    
    for trial in range(40):
        if time.time() > init_deadline:
            break
        spread = rng.random() < 0.3
        rows = _fast_greedy(k, n_rows, rng, spread=spread)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if not candidates:
        # Last resort: simple sequential
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                x = len(used) + j + 1
                while any((x - row[i]) in used for i in range(len(row))):
                    x += 1
                for i in range(len(row)):
                    used.add(x - row[i])
                row.append(x)
            rows.append(row)
        return rows
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Row rebuild with CSP to tighten initial solutions
    if time.time() < deadline - 40:
        rebuild_deadline = time.time() + min(15, deadline - time.time() - 40)
        for idx in range(min(3, len(candidates))):
            s, rows = candidates[idx]
            rebuild_scope, rebuild_rows = _row_rebuild_sa(rows, k, rng, rebuild_deadline, s)
            if rebuild_rows is not None and _valid(rebuild_rows, k):
                actual = _scope(rebuild_rows)
                if actual < best_scope:
                    best_scope = actual
                    best_rows = rebuild_rows
                    candidates.append((actual, rebuild_rows))
                    candidates.sort(key=lambda x: x[0])
    
    # Phase 3: SA optimization on top candidates
    remaining = deadline - time.time()
    if remaining > 10:
        top_n = min(3, len(candidates))
        time_per = remaining / max(1, top_n)
        
        for idx in range(top_n):
            if time.time() > deadline - 10:
                break
            s, rows = candidates[idx]
            sa_deadline = time.time() + time_per * 0.8
            sa_scope, sa_rows = _sa_optimize(rows, k, 50000, rng, sa_deadline)
            
            if sa_rows is not None and _valid(sa_rows, k):
                actual = _scope(sa_rows)
                if actual < best_scope:
                    best_scope = actual
                    best_rows = sa_rows
                    candidates.append((actual, sa_rows))
                    candidates.sort(key=lambda x: x[0])
    
    # Phase 4: Final polish on the best solution
    if best_rows is not None and time.time() < deadline - 3:
        polish_deadline = time.time() + min(8, deadline - time.time() - 3)
        polish_scope, polish_rows = _final_polish(best_rows, k, polish_deadline, rng)
        
        if polish_rows is not None and _valid(polish_rows, k):
            actual = _scope(polish_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = polish_rows
    
    return best_rows
