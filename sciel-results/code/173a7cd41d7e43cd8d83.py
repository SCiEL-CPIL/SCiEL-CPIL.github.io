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

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, rng, bias=1.0):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_x = last + 60
            while x <= max_x and len(candidates) < 10:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
            if not candidates:
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            weights = [1.0 / (c[0] - last + 1) ** bias for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        rows.append(row)
    return rows

class SAState:
    def __init__(self, rows, k, max_scope_bound=200):
        self.rows = [list(r) for r in rows]
        self.k = k
        self.n_rows = len(rows)
        self.max_scope_bound = max_scope_bound
        self.used_diffs = [0] * (max_scope_bound + 1)
        self.collisions = 0
        self.scope = 0
        self._rebuild_state()

    def _rebuild_state(self):
        self.used_diffs = [0] * (self.max_scope_bound + 1)
        self.collisions = 0
        self.scope = 0
        for r in self.rows:
            if r[-1] > self.scope:
                self.scope = r[-1]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= self.max_scope_bound:
                        self.used_diffs[d] += 1
        for d in range(1, self.max_scope_bound + 1):
            if self.used_diffs[d] > 1:
                self.collisions += self.used_diffs[d] - 1

    def get_cost(self):
        return self.scope + 1000 * self.collisions

    def try_move(self, i, j, new_val):
        row = self.rows[i]
        old_val = row[j]
        if new_val == old_val:
            return 0, True
        if j > 0 and new_val <= row[j - 1]:
            return 0, False
        if j < self.k and new_val >= row[j + 1]:
            return 0, False

        delta_coll = 0
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 1:
                    delta_coll -= 1
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 0:
                    delta_coll += 1

        for t in range(self.k + 1):
            if t == j:
                continue
            d_old = abs(row[t] - old_val)
            if d_old <= self.max_scope_bound:
                self.used_diffs[d_old] -= 1
            d_new = abs(row[t] - new_val)
            if d_new <= self.max_scope_bound:
                self.used_diffs[d_new] += 1

        self.collisions += delta_coll

        old_scope = self.scope
        if new_val > self.scope:
            self.scope = new_val
        elif old_val == self.scope and new_val < old_val:
            max_s = 0
            for r in self.rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            self.scope = max_s

        row[j] = new_val
        delta_cost = (self.scope - old_scope) + 1000 * delta_coll
        return delta_cost, True

    def revert_move(self, i, j, new_val, old_val):
        row = self.rows[i]
        delta_coll = 0
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 1:
                    delta_coll -= 1
                self.used_diffs[d] -= 1
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 0:
                    delta_coll += 1
                self.used_diffs[d] += 1
        self.collisions += delta_coll

        old_scope = self.scope
        if old_val > self.scope:
            self.scope = old_val
        elif new_val == self.scope and old_val < new_val:
            max_s = 0
            for r in self.rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            self.scope = max_s

        row[j] = old_val

def _sa_optimize(rows, k, deadline, rng, T0=10.0, cooling=0.9995, min_T=0.01):
    n_rows = len(rows)
    max_scope_bound = max(_scope(rows), 150) + 50
    state = SAState(rows, k, max_scope_bound)
    
    best_scope = state.scope
    best_rows = [list(r) for r in state.rows]
    
    T = T0
    penalty = 1000
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = state.rows[i][j]
        prev_val = state.rows[i][j - 1]
        next_val = state.rows[i][j + 1] if j < k else state.scope + 200
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        if rng.random() < 0.7 and old_val > lo:
            max_drop = old_val - lo
            drop = rng.randint(1, max(1, max_drop // 3))
            new_val = old_val - drop
        else:
            span = hi - lo
            if span <= 0:
                continue
            delta = int(rng.gauss(0, max(1, span * 0.2)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        delta_cost, success = state.try_move(i, j, new_val)
        
        if not success:
            T = max(T * cooling, min_T)
            continue
        
        if delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            if state.collisions == 0 and state.scope < best_scope:
                best_scope = state.scope
                best_rows = [list(r) for r in state.rows]
                if rng.random() < 0.03:
                    T = min(T * 1.5, 15.0)
        else:
            state.revert_move(i, j, new_val, old_val)
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _backtrack_construct(n_rows, k, max_scope, rng):
    used = set()
    rows = [None] * n_rows
    
    def backtrack(row_idx, col_idx, current_row):
        if row_idx == n_rows:
            return True
        
        if col_idx == 0:
            current_row = [0]
            return backtrack(row_idx, 1, current_row)
        
        if col_idx > k:
            rows[row_idx] = current_row
            row_diffs = _row_diffs(current_row)
            prev_used = set(used)
            used.update(row_diffs)
            
            if backtrack(row_idx + 1, 0, []):
                return True
            
            used.clear()
            used.update(prev_used)
            return False
        
        prev_val = current_row[-1]
        candidates = []
        for val in range(prev_val + 1, max_scope + 1):
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used:
                    valid = False
                    break
            if valid:
                candidates.append(val)
                if len(candidates) > 20:
                    break
        
        if not candidates:
            return False
        
        rng.shuffle(candidates)
        
        for val in candidates:
            current_row.append(val)
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            current_row.pop()
        
        return False
    
    if backtrack(0, 0, []):
        return rows
    return None

def _local_search(rows, k, deadline, rng):
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            diffs.update(_row_diffs(r))
        return diffs
    
    current_diffs = get_all_diffs(best_rows)
    
    for _ in range(100):
        if time.time() >= deadline:
            break
        
        idx = rng.randint(0, n_rows - 1)
        
        other_diffs = set()
        for i, r in enumerate(best_rows):
            if i != idx:
                other_diffs.update(_row_diffs(r))
        
        new_row = [0]
        local_diffs = set()
        
        for pos in range(1, k + 1):
            min_val = new_row[-1] + 1
            found = False
            
            search_end = min(best_scope, min_val + 30)
            
            candidates = list(range(min_val, search_end + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                valid = True
                new_diffs = []
                for a in new_row:
                    d = val - a
                    if d in other_diffs or d in local_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    new_row.append(val)
                    local_diffs.update(new_diffs)
                    found = True
                    break
            
            if not found:
                search_end = min(best_scope + 20, min_val + 50)
                candidates = list(range(min_val, search_end + 1))
                rng.shuffle(candidates)
                
                for val in candidates:
                    valid = True
                    new_diffs = []
                    for a in new_row:
                        d = val - a
                        if d in other_diffs or d in local_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        new_row.append(val)
                        local_diffs.update(new_diffs)
                        found = True
                        break
            
            if not found:
                break
        
        if len(new_row) == k + 1:
            new_scope = _scope([new_row] + [best_rows[i] for i in range(n_rows) if i != idx])
            if new_scope < best_scope:
                best_rows[idx] = new_row
                best_scope = new_scope
    
    return best_rows

def _sidon_precompute(n_rows, k, max_scope, rng, time_limit=15.0):
    """
    Pre-compute a set of valid Sidon rows (rows with distinct internal differences)
    sorted by their maximum entry. Returns a list of (max_entry, row, diff_set).
    """
    start = time.time()
    candidates = []
    seen_rows = set()
    
    # We want to find rows that are "tight" (small max entry)
    # A row is valid if all its differences are distinct.
    # We generate random rows and check validity.
    
    # To bias towards small max entries, we can try to fill rows greedily
    # with small numbers first, or just random sampling within a bound.
    
    # Let's use a hybrid: generate many random rows within [0, max_scope]
    # and keep the best ones.
    
    # Actually, for k=5 (6 elements), the number of differences is 15.
    # If max_scope is ~105, there are 105 possible differences.
    # The probability of a random set of 15 differences being distinct is decent.
    
    attempts = 0
    target_candidates = 5000
    
    while time.time() - start < time_limit and len(candidates) < target_candidates:
        # Generate a random increasing sequence of length k+1 starting at 0
        # within [0, max_scope]
        
        # Method: Randomly pick k points from 1..max_scope and sort them.
        # But we need to ensure they are distinct and >0.
        # Using random.sample is efficient.
        
        try:
            pts = sorted(rng.sample(range(1, max_scope + 1), k))
        except ValueError:
            continue
            
        row = [0] + pts
        
        # Check if it's a Sidon set (distinct differences)
        diffs = []
        valid = True
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.append(d)
            if not valid:
                break
        
        if valid:
            max_entry = row[-1]
            # We want to keep a diverse set, but specifically small max entries.
            # Let's just keep the best 5000 based on max_entry.
            # But we also want diversity.
            # For now, let's just collect them.
            
            # To avoid duplicates, we can use a tuple of the row as key.
            key = tuple(row)
            if key not in seen_rows:
                seen_rows.add(key)
                diff_set = set(diffs)
                candidates.append((max_entry, row, diff_set))
        
        attempts += 1
        
        # Early termination if we have enough "good" candidates
        # "Good" means max_entry <= some threshold
        # Let's say threshold is max_scope * 0.8
        # If we have 1000 candidates below threshold, stop.
        good_count = sum(1 for c in candidates if c[0] <= max_scope * 0.8)
        if good_count > 1000:
            break
    
    # Sort by max_entry
    candidates.sort(key=lambda x: x[0])
    return candidates

def _set_packing_construct(n_rows, k, candidates, rng, deadline):
    """
    Use the pre-computed Sidon rows to construct a full set via set packing.
    Greedily select rows that don't conflict with previously selected rows.
    Use randomization and backtracking.
    """
    # candidates is a list of (max_entry, row, diff_set)
    # We want to select n_rows rows such that their diff_sets are disjoint.
    
    # Since candidates are sorted by max_entry, we prioritize smaller ones.
    # But we need to ensure we can complete the set.
    
    # Strategy:
    # 1. Try to select the smallest possible rows.
    # 2. If stuck, backtrack.
    
    # To make it efficient, we can use a recursive backtracking search
    # that tries candidates in order of their max_entry.
    
    # However, the number of candidates might be large.
    # We can filter candidates to only those with max_entry <= current_best_scope.
    
    # Let's implement a simple backtracking search.
    
    best_solution = None
    best_scope = float('inf')
    
    # We need to be careful with time.
    # Let's limit the search depth or number of nodes.
    
    def backtrack(idx, used_diffs, current_rows, current_scope):
        nonlocal best_solution, best_scope
        
        if time.time() > deadline:
            return False
        
        if idx == n_rows:
            if current_scope < best_scope:
                best_scope = current_scope
                best_solution = [list(r) for r in current_rows]
            return True
        
        # Pruning: if current_scope >= best_scope, we can't improve (unless we find a smaller scope later? No, scope is max entry, so if we've already exceeded best_scope, we can stop this branch).
        # Actually, current_scope is the max entry of rows selected so far.
        # The final scope will be max(current_scope, max_entry_of_future_rows).
        # So if current_scope >= best_scope, we can prune.
        if current_scope >= best_scope:
            return False
            
        # Try candidates
        # We can iterate through candidates.
        # To speed up, we can skip candidates that conflict with used_diffs.
        
        # Optimization: Pre-filter candidates for this step?
        # Or just iterate and check.
        
        # To add randomness, we can shuffle the candidates or pick a random subset.
        # But since they are sorted by max_entry, we want to try small ones first.
        # Let's iterate from 0 to len(candidates).
        
        # To avoid trying the same candidate multiple times in different orders,
        # we should enforce an ordering? No, rows are distinct entities.
        # But if we swap two rows, the set of used_diffs is the same.
        # So we can enforce that we only pick candidates with index > last_picked_index?
        # This would treat rows as an unordered set, which is correct for the objective.
        # However, the "rows" in the output are just a list. The order doesn't matter for validity.
        
        # So we can do:
        # for i in range(start_idx, len(candidates)):
        
        # But we need to pass start_idx.
        
        # Let's modify backtrack to take a start index.
        
        return False # Placeholder, will be implemented below
    
    # Let's implement the actual backtracking with start index.
    
    def bt(idx, start_idx, used_diffs, current_rows, current_scope):
        nonlocal best_solution, best_scope
        
        if time.time() > deadline:
            return False
        
        if idx == n_rows:
            if current_scope < best_scope:
                best_scope = current_scope
                best_solution = [list(r) for r in current_rows]
            return True
        
        if current_scope >= best_scope:
            return False
            
        # Iterate candidates from start_idx
        # To improve performance, we can limit the number of candidates we try.
        # E.g., only try the first 100 valid candidates?
        
        tried = 0
        for i in range(start_idx, len(candidates)):
            max_entry, row, diff_set = candidates[i]
            
            # If max_entry is already larger than best_scope, we can break?
            # Not necessarily, because maybe this row is needed and others are small?
            # No, if max_entry >= best_scope, and we are trying to beat best_scope,
            # adding this row will make the final scope >= max_entry >= best_scope.
            # So we can break if candidates are sorted by max_entry.
            if max_entry >= best_scope:
                break
            
            # Check conflict
            conflict = False
            for d in diff_set:
                if d in used_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Try this row
            new_used = used_diffs | diff_set
            new_rows = current_rows + [row]
            new_scope = max(current_scope, max_entry)
            
            # Recurse
            if bt(idx + 1, i + 1, new_used, new_rows, new_scope):
                return True
                
            tried += 1
            if tried > 50: # Limit branching factor
                break
        
        return False
    
    bt(0, 0, set(), [], 0)
    
    if best_solution:
        return best_solution
    
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Initial Greedy Construction
    init_deadline = start_time + 5.0
    candidates = []
    
    while time.time() < init_deadline:
        bias = rng.uniform(0.5, 2.0)
        rows = _greedy_construct(n_rows, k, rng, bias=bias)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
        
        if best_scope < float('inf') and rng.random() < 0.3:
            target_scope = best_scope - 1
            if target_scope > n_rows * k:
                bt_rows = _backtrack_construct(n_rows, k, target_scope, rng)
                if bt_rows and _valid(bt_rows, k):
                    s = _scope(bt_rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [list(r) for r in bt_rows]
                    candidates.append((s, bt_rows))
    
    if not candidates:
        rows = _greedy_construct(n_rows, k, rng, bias=1.0)
        if _valid(rows, k):
            best_rows = rows
            best_scope = _scope(rows)
        else:
            return [[0, 1, 2, 3, 4, 5]] * n_rows
    
    # 2. Sidon Pre-computation and Set Packing
    # This is the structural change: use pre-computed valid rows and set packing.
    
    if time.time() < deadline - 20:
        # Pre-compute Sidon rows
        # We want to find rows with small max entries.
        # Let's aim for max_scope around best_scope - 10 to give some room.
        target_max_scope = max(best_scope - 5, 100)
        
        sidon_rows = _sidon_precompute(n_rows, k, target_max_scope, rng, time_limit=10.0)
        
        if sidon_rows:
            # Use set packing
            sp_deadline = time.time() + 20.0
            sp_rows = _set_packing_construct(n_rows, k, sidon_rows, rng, sp_deadline)
            
            if sp_rows and _valid(sp_rows, k):
                sp_scope = _scope(sp_rows)
                if sp_scope < best_scope:
                    best_scope = sp_scope
                    best_rows = sp_rows
    
    # 3. Refine with Local Search and SA
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:5]
    
    # Add the set packing result if it was better
    if best_rows:
        top_candidates.append((best_scope, best_rows))
        top_candidates.sort(key=lambda x: x[0])
        top_candidates = top_candidates[:5]
    
    remaining_time = deadline - time.time()
    if remaining_time <= 0:
        return best_rows
        
    time_per_sa = remaining_time / len(top_candidates)
    
    for i, (s, rows) in enumerate(top_candidates):
        if time.time() >= deadline - 5:
            break
        
        # First try local search (structural change)
        if time.time() < deadline - 10:
            local_deadline = time.time() + min(5.0, time_per_sa * 0.3)
            local_rows = _local_search(rows, k, local_deadline, rng)
            if _valid(local_rows, k):
                local_scope = _scope(local_rows)
                if local_scope < best_scope:
                    best_scope = local_scope
                    best_rows = local_rows
        
        # Then try simulated annealing
        sa_deadline = time.time() + time_per_sa * 0.6
        sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
        
        if _valid(sa_rows, k):
            actual_scope = _scope(sa_rows)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = sa_rows
                
                target_scope = best_scope - 1
                if target_scope > 0 and time.time() < deadline - 10:
                    bt_rows = _backtrack_construct(n_rows, k, target_scope, rng)
                    if bt_rows and _valid(bt_rows, k):
                        s2 = _scope(bt_rows)
                        if s2 < best_scope:
                            best_scope = s2
                            best_rows = [list(r) for r in bt_rows]
    
    return best_rows
