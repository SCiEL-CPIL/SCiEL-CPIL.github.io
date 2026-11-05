import random
import time
import sys
import math
import bisect

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave buffer for 120s limit (150s kill)
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row  # simple counting bound
    
    best_result = None
    best_scope = float('inf')
    
    def time_up():
        return time.time() > deadline
    
    def get_scope(rows):
        return max(r[-1] for r in rows)
    
    def get_used_diffs_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def _packing_score(new_diffs, used_sorted, x):
        """Score a candidate x by how well its differences pack into used space."""
        if not used_sorted:
            return x * 1.0
        total_dist = 0.0
        for d in new_diffs:
            idx = bisect.bisect_left(used_sorted, d)
            min_dist = float('inf')
            if idx < len(used_sorted):
                min_dist = min(min_dist, abs(d - used_sorted[idx]))
            if idx > 0:
                min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
            total_dist += min_dist
        return total_dist / max(len(new_diffs), 1) + 0.1 * x
    
    def _greedy_construct(n_rows, k, seed):
        """Best-fit greedy construction with packing heuristic."""
        rng = random.Random(seed)
        used = set()
        used_sorted = []
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                valid_xs = []
                scored_xs = []
                last_x = row[-1]
                limit = 500
                
                x = last_x + 1
                while len(valid_xs) < 40 and x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1
                
                if not valid_xs:
                    x = last_x + 1
                    while len(valid_xs) < 40:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            valid_xs.append(x)
                            score = _packing_score(new_diffs, used_sorted, x)
                            scored_xs.append((score, x))
                        x += 1
                
                if not valid_xs:
                    return None
                
                scored_xs.sort(key=lambda t: t[0])
                top_n = min(5, len(scored_xs))
                chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
                row.append(chosen_x)
                new_diffs = {chosen_x - a for a in row[:-1]}
                used.update(new_diffs)
                for d in new_diffs:
                    bisect.insort(used_sorted, d)
            
            rows.append(row)
        
        return rows
    
    def _simulated_annealing(initial_rows, n_rows, k):
        """
        Simulated annealing with O(1) diff-count validity checks.
        """
        nonlocal best_scope, best_result
        if initial_rows is None:
            return None
        
        rng = random.Random(42)
        rows = [list(r) for r in initial_rows]
        current_scope = get_scope(rows)
        best_local_scope = current_scope
        best_local_rows = [list(r) for r in rows]
        
        # Precompute difference sets for each row
        row_diffs = []
        for row in rows:
            n = len(row)
            diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.add(row[j] - row[i])
            row_diffs.append(diffs)
        
        # Maintain global diff count
        diff_count = {}
        for diffs in row_diffs:
            for d in diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
        
        def _row_is_valid(new_diffs, exclude_idx):
            for d in new_diffs:
                cnt = diff_count.get(d, 0)
                if d in row_diffs[exclude_idx]:
                    cnt -= 1
                if cnt > 0:
                    return False
            return True
        
        def _commit_row(row_idx, new_row, new_diffs):
            nonlocal rows, row_diffs
            for d in row_diffs[row_idx]:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            for d in new_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
            rows[row_idx] = new_row
            row_diffs[row_idx] = new_diffs
        
        def _compute_row_diffs(row):
            n = len(row)
            diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.add(row[j] - row[i])
            return diffs
        
        T_start = 20.0
        T_end = 0.3
        total_steps = 40000
        cool_rate = (T_end / T_start) ** (1.0 / total_steps)
        T = T_start
        steps_since_improve = 0
        max_no_improve = 4000
        
        for step in range(total_steps):
            if time_up():
                break
            if T < T_end:
                T = T_end
            
            # 40% of the time, target the row with the largest last element
            if rng.random() < 0.4:
                row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            else:
                row_idx = rng.randint(0, n_rows - 1)
            
            pos = rng.randint(1, k)
            row = rows[row_idx]
            old_val = row[pos]
            
            # Try to move element to a smaller value
            if pos > 0 and old_val > row[pos - 1] + 1:
                # Try random smaller values
                for _ in range(5):
                    new_val = rng.randint(row[pos - 1] + 1, old_val - 1)
                    if new_val == old_val:
                        continue
                    
                    # Compute new diffs for this row
                    new_row = row[:]
                    new_row[pos] = new_val
                    new_diffs = _compute_row_diffs(new_row)
                    
                    if _row_is_valid(new_diffs, row_idx):
                        # Accept with probability
                        delta_scope = max(new_row[-1], max(r[-1] for i, r in enumerate(rows) if i != row_idx)) - current_scope
                        if delta_scope <= 0 or rng.random() < math.exp(-delta_scope / T):
                            _commit_row(row_idx, new_row, new_diffs)
                            new_scope = get_scope(rows)
                            if new_scope < best_local_scope:
                                best_local_scope = new_scope
                                best_local_rows = [list(r) for r in rows]
                                steps_since_improve = 0
                            elif new_scope != current_scope:
                                current_scope = new_scope
                                steps_since_improve += 1
                            break
            else:
                steps_since_improve += 1
            
            T *= cool_rate
        
        if best_local_scope < best_scope:
            best_scope = best_local_scope
            best_result = [list(r) for r in best_local_rows]
        
        return best_local_rows
    
    def _polish_max(rows, n_rows, k):
        """Targeted local search to reduce the maximum element."""
        nonlocal best_scope, best_result
        if rows is None:
            return None
        
        current = [list(r) for r in rows]
        current_scope = get_scope(current)
        used = get_used_diffs_set(current)
        
        for _ in range(10000):
            if time_up():
                break
            
            # Find the row with the maximum last element
            max_row_idx = max(range(n_rows), key=lambda i: current[i][-1])
            max_row = current[max_row_idx]
            
            if max_row[-1] <= lower_bound:
                break
            
            improved = False
            # Try to reduce elements from the end backwards
            for pos in range(k, 0, -1):
                old_val = max_row[pos]
                if old_val <= max_row[pos - 1] + 1:
                    continue
                
                for new_val in range(old_val - 1, max_row[pos - 1], -1):
                    # Remove old differences involving this element
                    removed = set()
                    for i in range(len(max_row)):
                        if i != pos:
                            removed.add(abs(max_row[pos] - max_row[i]))
                    
                    # Tentatively update
                    max_row[pos] = new_val
                    new_diffs = set()
                    for i in range(len(max_row)):
                        if i != pos:
                            new_diffs.add(abs(max_row[pos] - max_row[i]))
                    
                    # Check conflicts
                    conflict = False
                    for d in new_diffs:
                        if d in used and d not in removed:
                            conflict = True
                            break
                    
                    if not conflict:
                        used -= removed
                        used |= new_diffs
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try swapping elements between rows
                found_swap = False
                for r1 in range(n_rows):
                    for r2 in range(r1 + 1, n_rows):
                        for p1 in range(1, len(current[r1])):
                            for p2 in range(1, len(current[r2])):
                                v1 = current[r1][p1]
                                v2 = current[r2][p2]
                                if v1 == v2:
                                    continue
                                
                                current[r1][p1] = v2
                                current[r2][p2] = v1
                                
                                if is_valid(current):
                                    new_scope = get_scope(current)
                                    if new_scope < current_scope:
                                        current_scope = new_scope
                                        used = get_used_diffs_set(current)
                                        improved = True
                                        found_swap = True
                                        break
                                else:
                                    current[r1][p1] = v1
                                    current[r2][p2] = v2
                            
                            if found_swap:
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break
            
            if not improved:
                break
        
        final_scope = get_scope(current)
        if final_scope < best_scope:
            best_scope = final_scope
            best_result = [list(r) for r in current]
        
        return current
    
    def solve_attempt(target_scope, rng_seed):
        """Backtracking with bitmask for small differences."""
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            upper_bound = target_scope - 1
            
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d >= 256:
                        continue
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                noise = rng.random() * 0.01
                candidates.append((cost + noise, x, new_mask))
                
                if len(candidates) > 15:
                    break
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: item[0])
            
            top_n = min(4, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining_candidates = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining_candidates:
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0])
    
    # Phase 1: Initial greedy construction
    seed = 0
    while time.time() < deadline - 70:
        rows = _greedy_construct(n_rows, k, seed)
        if rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_result = [list(r) for r in rows]
                if best_scope == lower_bound:
                    break
        seed += 1
    
    if best_result is None:
        best_result = _greedy_construct(n_rows, k, 0)
        best_scope = get_scope(best_result)
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 20:
        improved = _simulated_annealing(best_result, n_rows, k)
        if improved is not None:
            imp_scope = get_scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_result = [list(r) for r in improved]
    
    # Phase 3: Backtracking attempts
    seed_counter = 0
    while time.time() < deadline - 15.0:
        if best_result is not None:
            target = best_scope
        else:
            target = 200
        
        solve_attempt(target, seed_counter)
        seed_counter += 1
        
        if seed_counter % 3 == 0 and best_result is not None:
            _polish_max(best_result, n_rows, k)
    
    # Phase 4: Final polish
    if best_result is not None and time.time() < deadline - 2.0:
        _polish_max(best_result, n_rows, k)
    
    if best_result is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_result
