import time
import random
import math

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

def _diffs_of_row(row):
    """Get all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Get all differences across all rows."""
    seen = set()
    for r in rows:
        seen |= _diffs_of_row(r)
    return seen

def _greedy_init(n_rows, k, max_scope, rng):
    """Greedy construction with forward checking."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            found = False
            for x in range(last + 1, max_scope + 1):
                ok = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    row.append(x)
                    used.update(new_diffs)
                    last = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _dfs_construct(n_rows, k, rng, max_scope=120):
    """DFS construction with pruning."""
    used = set()
    rows = []
    
    def backtrack(row_idx, pos, current_row, last_val):
        if pos == k + 1:
            rows.append(list(current_row))
            return True
        
        # Try candidates in order of preference (smaller first, with some randomization)
        candidates = []
        for v in range(last_val + 1, min(last_val + 40, max_scope + 1)):
            valid = True
            new_diffs = []
            for i in range(len(current_row)):
                d = v - current_row[i]
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                candidates.append((v, new_diffs))
        
        if not candidates:
            return False
        
        # Randomize slightly
        if rng.random() < 0.3:
            rng.shuffle(candidates)
        
        for v, new_diffs in candidates:
            current_row.append(v)
            used.update(new_diffs)
            
            if backtrack(row_idx, pos + 1, current_row, v):
                return True
            
            current_row.pop()
            used.difference_update(new_diffs)
        
        return False
    
    for r in range(n_rows):
        if not backtrack(r, 1, [0], 0):
            return None
    return rows

class DiffTracker:
    """Efficient difference tracking for SA."""
    def __init__(self, rows, k):
        self.rows = [list(r) for r in rows]
        self.k = k
        self.n_rows = len(rows)
        self.used_diffs = set()
        self._rebuild()
    
    def _rebuild(self):
        self.used_diffs = set()
        for row in self.rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    self.used_diffs.add(row[j] - row[i])
    
    def scope(self):
        return max(a for r in self.rows for a in r[1:])
    
    def try_move(self, row_idx, pos, new_val):
        """Try to change rows[row_idx][pos] to new_val. Return (delta_scope, success)."""
        row = self.rows[row_idx]
        old_val = row[pos]
        if new_val == old_val:
            return 0, True
        if new_val <= row[pos-1] or (pos < len(row)-1 and new_val >= row[pos+1]):
            return 0, False
        
        # Remove old diffs
        removed = []
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(row[pos] - row[t])
            removed.append(d)
            self.used_diffs.discard(d)
        
        # Check conflicts
        conflicts = 0
        new_diffs = []
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(new_val - row[t])
            if d in self.used_diffs:
                conflicts += 1
            new_diffs.append(d)
        
        if conflicts > 0:
            for d in removed:
                self.used_diffs.add(d)
            return 0, False
        
        # Apply
        row[pos] = new_val
        old_scope = self.scope()
        for d in new_diffs:
            self.used_diffs.add(d)
        new_scope = self.scope()
        return new_scope - old_scope, True
    
    def revert_move(self, row_idx, pos, old_val):
        """Revert move."""
        row = self.rows[row_idx]
        new_val = row[pos]
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(new_val - row[t])
            self.used_diffs.discard(d)
        row[pos] = old_val
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(old_val - row[t])
            self.used_diffs.add(d)

def _sa_optimize(rows, k, deadline, rng, T0=6.0, cooling=0.9998, min_T=0.2):
    """Simulated annealing with restarts."""
    tracker = DiffTracker(rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = tracker.scope()
    
    T = T0
    no_improve = 0
    restart_thresh = 2000
    
    while time.time() < deadline:
        row_idx = rng.randint(0, tracker.n_rows - 1)
        pos = rng.randint(1, tracker.k)
        row = tracker.rows[row_idx]
        old_val = row[pos]
        
        # Propose move
        if rng.random() < 0.75:
            max_drop = old_val - row[pos-1] - 1
            if max_drop <= 0:
                T = max(T * cooling, min_T)
                continue
            drop = rng.randint(1, max_drop)
            new_val = old_val - drop
        else:
            delta = rng.randint(1, 6)
            new_val = old_val + delta
        
        lo = row[pos-1] + 1
        hi = row[pos+1] - 1 if pos < tracker.k else 200
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            T = max(T * cooling, min_T)
            continue
        
        delta, success = tracker.try_move(row_idx, pos, new_val)
        if not success:
            T = max(T * cooling, min_T)
            continue
        
        if delta <= 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
            if tracker.scope() < best_scope:
                best_scope = tracker.scope()
                best_rows = [list(r) for r in tracker.rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            tracker.revert_move(row_idx, pos, old_val)
            no_improve += 1
        
        T = max(T * cooling, min_T)
        
        if no_improve > restart_thresh:
            tracker.rows = [list(r) for r in best_rows]
            tracker._rebuild()
            T = T0
            no_improve = 0
    
    return best_scope, best_rows

def _local_search(rows, k, deadline, rng):
    """Iterative local search to minimize scope."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    used = _all_diffs(best_rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            for j in range(k, 0, -1):
                lo = row[j-1] + 1
                for new_val in range(lo, row[j]):
                    old_val = row[j]
                    
                    removed = []
                    for t in range(k+1):
                        if t == j:
                            continue
                        d = abs(row[j] - row[t])
                        removed.append(d)
                        used.discard(d)
                    
                    ok = True
                    new_diffs = []
                    for t in range(k+1):
                        if t == j:
                            continue
                        d = abs(new_val - row[t])
                        if d in used:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if ok:
                        row[j] = new_val
                        used.update(new_diffs)
                        ns = _scope(best_rows)
                        if ns < best_scope:
                            best_scope = ns
                            improved = True
                        else:
                            row[j] = old_val
                            used.difference_update(new_diffs)
                            used.update(removed)
                        break
                    else:
                        for d in removed:
                            used.add(d)
    
    return best_scope, best_rows

def _branch_bound_find_clique(n_rows, k, max_scope, rng, deadline):
    """
    Branch and bound: find n_rows disjoint difference sets, each from a valid row.
    Each row corresponds to a set of C(k+1,2) differences.
    We want to minimize the maximum value.
    
    Strategy: enumerate possible rows (as difference sets) in order of max value,
    then find a clique of size n_rows in the intersection graph.
    """
    # This is a simplified version - we'll use a heuristic approach
    # Generate candidate rows and find compatible sets
    
    candidates_by_scope = {}
    
    # Generate diverse candidate rows
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Generate a random valid row
        row = [0]
        last = 0
        valid = True
        for j in range(1, k + 1):
            # Try random values
            found = False
            for attempt in range(20):
                x = last + rng.randint(1, 15)
                if x > max_scope:
                    break
                ok = True
                for i in range(len(row)):
                    d = x - row[i]
                    if d <= 0:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    last = x
                    found = True
                    break
            if not found:
                valid = False
                break
        
        if valid and len(row) == k + 1:
            diffs = _diffs_of_row(row)
            scope = row[-1]
            if scope <= max_scope:
                if scope not in candidates_by_scope:
                    candidates_by_scope[scope] = []
                # Check uniqueness
                row_tuple = tuple(row)
                if row_tuple not in candidates_by_scope[scope]:
                    candidates_by_scope[scope].append(row)
    
    # Now find a set of n_rows mutually compatible rows
    # Using greedy + local search
    best_set = None
    best_max_scope = float('inf')
    
    # Sort scopes
    scopes = sorted(candidates_by_scope.keys())
    
    for target_scope in scopes:
        if time.time() > deadline:
            break
        
        # Get all rows with scope <= target_scope
        all_rows = []
        for s in scopes:
            if s > target_scope:
                break
            all_rows.extend(candidates_by_scope[s])
        
        if len(all_rows) < n_rows:
            continue
        
        # Greedy: pick rows that are mutually compatible
        used_diffs = set()
        selected = []
        remaining = list(all_rows)
        rng.shuffle(remaining)
        
        for row in remaining:
            if len(selected) >= n_rows:
                break
            diffs = _diffs_of_row(row)
            if not diffs & used_diffs:
                selected.append(row)
                used_diffs |= diffs
        
        if len(selected) == n_rows:
            actual_scope = max(r[-1] for r in selected)
            if actual_scope < best_max_scope:
                best_max_scope = actual_scope
                best_set = [list(r) for r in selected]
    
    return best_set, best_max_scope

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions
    init_deadline = start_time + 25.0
    candidates = []
    
    # Try different construction methods
    for max_s in [108, 110, 112, 115, 120]:
        if time.time() > init_deadline:
            break
        
        # DFS construction
        for trial in range(5):
            if time.time() > init_deadline:
                break
            rows = _dfs_construct(n_rows, k, rng, max_scope=max_s)
            if rows is not None and _valid(rows, k):
                s = _scope(rows)
                candidates.append((s, rows))
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
        
        # Greedy construction
        for trial in range(3):
            if time.time() > init_deadline:
                break
            rows = _greedy_init(n_rows, k, max_s, rng)
            if rows is not None and _valid(rows, k):
                s = _scope(rows)
                candidates.append((s, rows))
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
    
    # Try branch-and-bound approach
    if time.time() < init_deadline:
        bb_deadline = min(init_deadline, time.time() + 15)
        bb_rows, bb_scope = _branch_bound_find_clique(n_rows, k, 125, rng, bb_deadline)
        if bb_rows is not None and _valid(bb_rows, k):
            s = _scope(bb_rows)
            candidates.append((s, bb_rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in bb_rows]
    
    if not candidates:
        # Last resort
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
    
    # Phase 2: SA optimization on top candidates
    remaining = deadline - time.time()
    if remaining > 10:
        top_n = min(5, len(candidates))
        time_per = remaining * 0.65 / max(1, top_n)
        
        for idx in range(top_n):
            if time.time() > deadline - 15:
                break
            s, rows = candidates[idx]
            sa_deadline = time.time() + time_per
            sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
            
            if sa_rows is not None and _valid(sa_rows, k):
                actual = _scope(sa_rows)
                if actual < best_scope:
                    best_scope = actual
                    best_rows = sa_rows
                    candidates.append((actual, sa_rows))
                    candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Final polish
    if best_rows is not None and time.time() < deadline - 8:
        polish_deadline = time.time() + min(10, deadline - time.time() - 5)
        _, polished = _local_search(best_rows, k, polish_deadline, rng)
        if _valid(polished, k):
            ps = _scope(polished)
            if ps < best_scope:
                best_scope = ps
                best_rows = polished
    
    return best_rows
