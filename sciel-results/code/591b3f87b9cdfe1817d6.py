import time
import random
import math

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers)
    # We want to minimize M (max entry)
    
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    def edges_from_row(row):
        """Get edge vectors (consecutive differences) from a row."""
        return [row[i+1] - row[i] for i in range(len(row)-1)]

    def row_from_edges(edges):
        """Reconstruct row from edge vectors."""
        row = [0]
        for e in edges:
            row.append(row[-1] + e)
        return row

    def all_diffs_from_edges(edges):
        """Compute all positive differences from edge vectors.
        For edges e_1, e_2, ..., e_k, the differences are all sums of consecutive edges."""
        diffs = set()
        n = len(edges)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += edges[j]
                diffs.add(s)
        return diffs

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    def greedy_interleaved():
        """Interleaved greedy: add one point to the row with the smallest current max,
        breaking ties by smallest last element. This is the 'bin-packing' heuristic."""
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for _ in range(n_rows * k):
            # Pick row with smallest last element (most room), tie-break by smallest max
            idx = min(range(n_rows), key=lambda i: (rows[i][-1], max(rows[i])))
            row = rows[idx]
            x = row[-1] + 1
            while True:
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
                    break
                x += 1
        return rows

    # ---------- Lattice-based edge vector construction ----------
    def build_row_from_edges(used_other, max_val=200, jitter=0.0, max_attempts=50):
        """Build a row using lattice-based edge vector selection."""
        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            
            edges = []
            row_diffs = set()
            success = True
            
            for pos in range(k):
                candidates = []
                for e in range(1, max_val):
                    new_diffs = set()
                    new_diffs.add(e)
                    s = e
                    for prev_pos in range(pos - 1, -1, -1):
                        s += edges[prev_pos]
                        new_diffs.add(s)
                    
                    if new_diffs & used_other:
                        continue
                    if new_diffs & row_diffs:
                        continue
                    
                    candidates.append(e)
                
                if not candidates:
                    success = False
                    break
                
                if jitter > 0 and len(candidates) > 1:
                    weights = [math.exp(-c / (20 + 5 * jitter * 10)) for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen_idx = i
                            break
                    chosen_e = candidates[chosen_idx]
                else:
                    chosen_e = candidates[0]
                
                edges.append(chosen_e)
                new_diffs = set()
                new_diffs.add(chosen_e)
                s = chosen_e
                for prev_pos in range(pos - 1, -1, -1):
                    s += edges[prev_pos]
                    new_diffs.add(s)
                row_diffs |= new_diffs
            
            if success:
                row = row_from_edges(edges)
                if max(row) < max_val and not (all_diffs_from_edges(edges) & used_other):
                    return row, edges
        
        return None, None

    def greedy_edges(used_other, max_val=300):
        """Deterministic greedy edge-based row construction."""
        edges = []
        row_diffs = set()
        
        for pos in range(k):
            found = False
            for e in range(1, max_val):
                new_diffs = set()
                new_diffs.add(e)
                s = e
                for prev_pos in range(pos - 1, -1, -1):
                    s += edges[prev_pos]
                    new_diffs.add(s)
                
                if not (new_diffs & used_other) and not (new_diffs & row_diffs):
                    edges.append(e)
                    row_diffs |= new_diffs
                    found = True
                    break
            
            if not found:
                return None
        
        return row_from_edges(edges)

    # ---------- Backtracking solver with forward checking (structural change) ----------
    def backtracking_solver(M, time_budget=20.0, randomize=False, seed=None):
        """Backtracking search for a valid set of n_rows rows with max entry <= M.
        Uses forward checking and dynamic variable ordering (most-constrained row first).
        Returns (cost, rows) or None."""
        if seed is not None:
            random.seed(seed)
        
        row_len = k + 1
        start_t = time.time()
        bt_deadline = start_t + time_budget
        
        def bt_time_left():
            return bt_deadline - time.time()
        
        # State: rows (list of lists), used (set of all diffs used)
        rows = [[0] for _ in range(n_rows)]
        used = set()
        
        # Precompute which positions in a row are "next" for each row
        # We track: for each row, how many elements it has
        # Variable ordering: pick row with fewest elements, tie-break by smallest last element
        
        def order_rows():
            """Return indices of rows ordered by constraint (fewest elements first)."""
            return sorted(range(n_rows), key=lambda i: (len(rows[i]), rows[i][-1] if rows[i] else 0))
        
        def find_conflict(new_val, row, used_set):
            """Check if adding new_val to row creates conflicts. Return None if ok, else conflicting diff."""
            for v in row:
                d = new_val - v
                if d in used_set:
                    return d
            return None
        
        # Backtracking with iterative deepening on M
        # We'll do DFS with forward checking
        
        best_result = [None]
        nodes_visited = [0]
        
        def dfs(depth):
            """Recursive DFS. depth = total points placed so far."""
            if bt_time_left() < 0.5:
                return False
            
            nodes_visited[0] += 1
            
            # Check if complete
            if all(len(r) == row_len for r in rows):
                best_result[0] = [list(r) for r in rows]
                return True
            
            # Pick most-constrained row
            # Rows that aren't full, ordered by (len, last_element)
            candidates_rows = [i for i in range(n_rows) if len(rows[i]) < row_len]
            if not candidates_rows:
                return True  # all full
            
            # Most constrained: fewest elements, then smallest last (most room)
            idx = min(candidates_rows, key=lambda i: (len(rows[i]), rows[i][-1]))
            current_row = rows[idx]
            
            # Try candidate values
            x = current_row[-1] + 1
            max_x = M
            
            # Generate candidates in order
            if randomize and random.random() < 0.3:
                # Randomized: pick a random subset to explore
                all_cands = []
                while x <= max_x:
                    conflict = find_conflict(x, current_row, used)
                    if conflict is None:
                        all_cands.append(x)
                    x += 1
                if not all_cands:
                    return False
                # Shuffle but keep some structure
                random.shuffle(all_cands)
                # Limit to first 50 for speed
                all_cands = all_cands[:50]
                for new_val in all_cands:
                    # Forward check: add to row and used
                    rows[idx].append(new_val)
                    new_diffs = {new_val - v for v in current_row}
                    used.update(new_diffs)
                    
                    if dfs(depth + 1):
                        return True
                    
                    # Undo
                    used.difference_update(new_diffs)
                    rows[idx].pop()
            else:
                # Deterministic: try smallest first
                count = 0
                while x <= max_x:
                    conflict = find_conflict(x, current_row, used)
                    if conflict is None:
                        count += 1
                        # Forward check
                        rows[idx].append(x)
                        new_diffs = {x - v for v in current_row}
                        used.update(new_diffs)
                        
                        if dfs(depth + 1):
                            return True
                        
                        # Undo
                        used.difference_update(new_diffs)
                        rows[idx].pop()
                    x += 1
                    if count > 80:  # Limit branching
                        break
            
            return False
        
        success = dfs(0)
        if success and best_result[0] is not None:
            return (M, best_result[0])
        return None
    
    def iterative_deepening_backtrack(M_start, M_max, time_budget=30.0):
        """Iterative deepening: try M_start, M_start+1, ... until success or time runs out."""
        start_t = time.time()
        id_deadline = start_t + time_budget
        
        M = M_start
        while M <= M_max:
            if time.time() - start_t > time_budget:
                break
            remaining = id_deadline - time.time()
            if remaining < 3:
                break
            
            # Try deterministic first, then randomized
            result = backtracking_solver(M, time_budget=min(remaining * 0.5, 10.0), randomize=False)
            if result is not None:
                return result
            
            if time.time() - start_t > time_budget * 0.7:
                break
            
            # Try randomized
            result = backtracking_solver(M, time_budget=min(remaining * 0.3, 8.0), randomize=True, seed=int(time.time() * 1000) % 100000)
            if result is not None:
                return result
            
            M += 1
        
        return None

    # ---------- Per-row local optimization (refinement machinery) ----------
    def try_reduce_row(rows, row_idx, used_other, max_attempts=150):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            
            if attempt % 3 == 0:
                jitter = 0.1 + 0.1 * (attempt % 10)
                max_val = min(current_max - 1, 250)
                new_row, _ = build_row_from_edges(used_other, max_val=max_val, jitter=jitter, max_attempts=10)
            elif attempt % 3 == 1:
                row, x = [0], 0
                current_row_diffs = set()
                success = True
                jitter = 0.15 if attempt > 15 else 0.0
                while len(row) < k + 1:
                    found = False
                    candidates = []
                    for trial_x in range(x + 1, x + 80):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            score = trial_x
                            candidates.append((score, trial_x, new))
                    
                    if not candidates:
                        success = False
                        break
                    
                    candidates.sort(key=lambda c: c[0])
                    
                    chosen_idx = 0
                    if jitter > 0 and len(candidates) > 1:
                        if random.random() < jitter:
                            top_n = min(5, len(candidates))
                            chosen_idx = random.randint(0, top_n - 1)
                    
                    _, chosen_x, chosen_new = candidates[chosen_idx]
                    row.append(chosen_x)
                    current_row_diffs |= chosen_new
                    x = row[-1]
                
                if success and max(row) < best_max:
                    test_rows = [list(r) for r in rows]
                    test_rows[row_idx] = row
                    if is_valid(test_rows):
                        best_row = list(row)
                        best_max = max(row)
            else:
                jitter = 0.2 + 0.05 * (attempt % 8)
                max_val = min(current_max, 280)
                new_row, _ = build_row_from_edges(used_other, max_val=max_val, jitter=jitter, max_attempts=15)
            
            if new_row is not None and max(new_row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = new_row
                if is_valid(test_rows):
                    best_row = list(new_row)
                    best_max = max(new_row)
        
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 6
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=80)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            success = False
            for method in range(2):
                if method == 0:
                    for attempt in range(20):
                        jitter = 0.3 + 0.1 * random.random()
                        max_val = 350
                        new_row, _ = build_row_from_edges(used_other, max_val=max_val, jitter=jitter, max_attempts=5)
                        if new_row is not None:
                            rows[idx] = new_row
                            success = True
                            break
                else:
                    for attempt in range(25):
                        row, x = [0], 0
                        current_row_diffs = set()
                        valid_row = True
                        while len(row) < k + 1:
                            found = False
                            candidates = []
                            for trial_x in range(x + 1, x + 120):
                                new = {trial_x - a for a in row}
                                if not (new & used_other) and not (new & current_row_diffs):
                                    score = trial_x + random.random() * 20
                                    candidates.append((score, trial_x, new))
                            
                            if not candidates:
                                valid_row = False
                                break
                            
                            candidates.sort(key=lambda c: c[0])
                            
                            if len(candidates) > 1 and random.random() < 0.45:
                                top_n = min(10, len(candidates))
                                chosen_idx = random.randint(0, top_n - 1)
                            else:
                                chosen_idx = 0
                            
                            _, chosen_x, chosen_new = candidates[chosen_idx]
                            row.append(chosen_x)
                            current_row_diffs |= chosen_new
                            x = row[-1]
                        
                        if valid_row:
                            rows[idx] = row
                            success = True
                            break
                
                if success:
                    break
            
            if not success:
                greedy_row = greedy_edges(used_other, max_val=400)
                if greedy_row is None:
                    greedy_row, x = [0], 0
                    row_diffs = set()
                    while len(greedy_row) < k + 1:
                        x += 1
                        new = {x - a for a in greedy_row}
                        if not (new & used_other) and not (new & row_diffs):
                            greedy_row.append(x)
                            row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # ---------- Main search loop ----------
    # Start with both greedy baselines and pick the better one
    rows_seq = greedy()
    rows_int = greedy_interleaved()
    best_rows = rows_seq if scope(rows_seq) <= scope(rows_int) else rows_int
    best_scope = scope(best_rows)

    # Initial polish to get a tighter baseline
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Phase 1: Iterative deepening backtracking to find the optimal (or near-optimal) scope
    # Try to find a solution with scope = best_scope - 1, best_scope - 2, etc.
    # But first, try to confirm best_scope is achievable (it is, by construction)
    
    # Use backtracking to try to find a smaller scope
    # Start searching from best_scope - 1 downward... actually upward from a lower bound
    # Lower bound: counting argument gives ~112, but let's start from best_scope - 5
    
    if time_left() > 40:
        # Try to find a solution with scope less than best_scope
        # Use iterative deepening: try M = best_scope - 1, best_scope - 2, ...
        # But that's expensive. Instead, do a few targeted backtracking searches.
        
        # First, try a direct backtracking search with M = best_scope - 1
        M_try = best_scope - 1
        while M_try >= max(112, best_scope - 20) and time_left() > 25:
            remaining = time_left() - 10
            if remaining < 5:
                break
            
            # Deterministic backtracking
            bt_result = backtracking_solver(M_try, time_budget=min(remaining * 0.5, 15.0), randomize=False)
            if bt_result is not None:
                bt_cost, bt_rows = bt_result
                if bt_cost < best_scope:
                    best_rows = bt_rows
                    best_scope = bt_cost
                    # Polish
                    if time_left() > 15:
                        polished_bt = local_polish(best_rows)
                        if scope(polished_bt) < best_scope:
                            best_rows = polished_bt
                            best_scope = scope(polished_bt)
                    break
            
            # Randomized backtracking
            if time_left() > 15:
                rand_seed = int(time.time() * 1000) % 100000
                bt_result = backtracking_solver(M_try, time_budget=min(remaining * 0.3, 10.0), randomize=True, seed=rand_seed)
                if bt_result is not None:
                    bt_cost, bt_rows = bt_result
                    if bt_cost < best_scope:
                        best_rows = bt_rows
                        best_scope = bt_cost
                        if time_left() > 15:
                            polished_bt = local_polish(best_rows)
                            if scope(polished_bt) < best_scope:
                                best_rows = polished_bt
                                best_scope = scope(polished_bt)
                        break
            
            M_try -= 1

    # Phase 2: ILS with backtracking-based kicks
    kick_count = 0
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        kick_count += 1
        
        # 2. Every few kicks, try backtracking from current best to find smaller scope
        if kick_count % 3 == 0 and time_left() > 30:
            M_try = best_scope - 1
            found_better = False
            while M_try >= max(112, best_scope - 10) and time_left() > 20:
                remaining = time_left() - 10
                if remaining < 5:
                    break
                
                bt_result = backtracking_solver(M_try, time_budget=min(remaining * 0.4, 12.0), randomize=False)
                if bt_result is not None:
                    bt_cost, bt_rows = bt_result
                    if bt_cost < best_scope:
                        best_rows = bt_rows
                        best_scope = bt_cost
                        if time_left() > 15:
                            polished_bt = local_polish(best_rows)
                            if scope(polished_bt) < best_scope:
                                best_rows = polished_bt
                                best_scope = scope(polished_bt)
                        found_better = True
                        break
                
                if time_left() > 15:
                    rand_seed = int(time.time() * 1000) % 100000
                    bt_result = backtracking_solver(M_try, time_budget=min(remaining * 0.25, 8.0), randomize=True, seed=rand_seed)
                    if bt_result is not None:
                        bt_cost, bt_rows = bt_result
                        if bt_cost < best_scope:
                            best_rows = bt_rows
                            best_scope = bt_cost
                            if time_left() > 15:
                                polished_bt = local_polish(best_rows)
                                if scope(polished_bt) < best_scope:
                                    best_rows = polished_bt
                                    best_scope = scope(polished_bt)
                            found_better = True
                            break
                
                M_try -= 1
            
            if found_better:
                continue
        
        # 3. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 4. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final high-precision polish pass
    if time_left() > 8:
        final_polish = local_polish(best_rows)
        if scope(final_polish) < best_scope:
            best_rows = final_polish
            best_scope = scope(final_polish)

    # Final backtracking attempt to squeeze out last units
    if time_left() > 15:
        M_try = best_scope - 1
        while M_try >= max(112, best_scope - 5) and time_left() > 10:
            remaining = time_left() - 5
            if remaining < 3:
                break
            bt_result = backtracking_solver(M_try, time_budget=remaining * 0.8, randomize=False)
            if bt_result is not None:
                bt_cost, bt_rows = bt_result
                if bt_cost < best_scope:
                    best_rows = bt_rows
                    best_scope = bt_cost
                    break
            M_try -= 1

    return best_rows
