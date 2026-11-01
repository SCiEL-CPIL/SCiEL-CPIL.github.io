import time
import math
import random
import sys

def construct(n_rows, k):
    """
    Construct difference triangle set (n_rows, k).
    Returns a list of n_rows rows, each of length k+1, starting at 0, strictly increasing.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit
    
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    M_lower = total_diffs_needed
    
    # Greedy construction for initial upper bound
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows
    
    greedy_rows = greedy_construct()
    best_scope = max(max(r) for r in greedy_rows)
    best_rows = [r[:] for r in greedy_rows]
    
    # Conflict-driven backtracking with multi-start
    def compute_all_diffs(rows):
        """Return dict: diff -> list of (row_idx, i, j) producing it"""
        diff_map = {}
        for ri, row in enumerate(rows):
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d not in diff_map:
                        diff_map[d] = []
                    diff_map[d].append((ri, i, j))
        return diff_map
    
    def get_conflicts(diff_map):
        """Return list of (diff, count) for diffs with count > 1"""
        return [(d, len(v)) for d, v in diff_map.items() if len(v) > 1]
    
    def count_conflicts(diff_map):
        return sum(1 for v in diff_map.values() if len(v) > 1)
    
    # Key insight: for each element, precompute which diffs it participates in
    # and use a priority queue of most-conflicting elements to move first
    
    def targeted_backtrack(rows, max_scope, time_limit):
        """
        Try to resolve all conflicts by moving elements that participate in most conflicts.
        Uses a greedy approach: repeatedly pick the most-conflicting element and
        try to reposition it to reduce conflicts.
        """
        rows = [r[:] for r in rows]
        diff_map = compute_all_diffs(rows)
        n_conflicts = count_conflicts(diff_map)
        
        if n_conflicts == 0:
            return rows, 0
        
        # Track which elements participate in which conflicts
        # element_key = (ri, ei)
        elem_conflict_count = {}
        
        def update_elem_conflicts(diff_map):
            elem_conflict_count.clear()
            for d, positions in diff_map.items():
                if len(positions) > 1:
                    for (ri, i, j) in positions:
                        key1 = (ri, i)
                        key2 = (ri, j)
                        elem_conflict_count[key1] = elem_conflict_count.get(key1, 0) + 1
                        elem_conflict_count[key2] = elem_conflict_count.get(key2, 0) + 1
            return elem_conflict_count
        
        elem_conflict_count = update_elem_conflicts(diff_map)
        
        # Maintain a list of candidate moves: for each element, possible new positions
        # that would reduce its conflict count
        
        best_local = [r[:] for r in rows]
        best_local_conflicts = n_conflicts
        
        max_iter = 20000
        iter_count = 0
        
        while iter_count < max_iter and time.time() < time_limit:
            if n_conflicts == 0:
                break
            
            # Pick the element with highest conflict involvement
            if not elem_conflict_count:
                break
            
            # Find element with max conflict count
            max_conf = max(elem_conflict_count.values())
            candidates = [key for key, cnt in elem_conflict_count.items() if cnt == max_conf]
            ri, ei = random.choice(candidates)
            
            row = rows[ri]
            old_val = row[ei]
            
            # Determine valid range
            lower = row[ei - 1] + 1 if ei > 0 else 1
            upper = row[ei + 1] - 1 if ei < k else max_scope
            
            if lower > upper:
                # Try to expand scope slightly
                if ei == k:
                    upper = min(max_scope + 5, best_scope + 20)
                else:
                    iter_count += 1
                    continue
            
            # Try several candidate positions, prioritizing those that reduce conflicts
            # Sample a few random positions in range
            num_tries = min(20, upper - lower + 1)
            best_new_val = old_val
            best_new_conflicts = n_conflicts
            
            for _ in range(num_tries):
                if upper - lower > 10:
                    new_val = random.randint(lower, upper)
                else:
                    new_val = random.randint(lower, upper)
                
                if new_val == old_val:
                    continue
                
                # Quick conflict estimate: count how many of this element's current
                # conflicts would be resolved
                row = rows[ri]
                old_diffs = set()
                for j in range(len(row)):
                    if j != ei:
                        d = abs(row[j] - old_val)
                        old_diffs.add(d)
                
                new_diffs = set()
                for j in range(len(row)):
                    if j != ei:
                        d = abs(row[j] - new_val)
                        new_diffs.add(d)
                
                # Estimate: conflicts removed = diffs in old_diffs that are currently
                # conflicted and not in new_diffs
                # Conflicts added = diffs in new_diffs that are currently unconflicted
                # This is an approximation; we'll do exact check for best candidates
                
                # Exact check: simulate the move
                saved_row = row[:]
                row[ei] = new_val
                new_diff_map = compute_all_diffs(rows)
                new_conflicts = count_conflicts(new_diff_map)
                row[:] = saved_row
                
                if new_conflicts < best_new_conflicts:
                    best_new_conflicts = new_conflicts
                    best_new_val = new_val
            
            if best_new_val != old_val and best_new_conflicts < n_conflicts:
                # Apply the move
                row[ei] = best_new_val
                diff_map = compute_all_diffs(rows)
                n_conflicts = best_new_conflicts
                elem_conflict_count = update_elem_conflicts(diff_map)
                
                if n_conflicts == 0:
                    best_local = [r[:] for r in rows]
                    break
                elif n_conflicts < best_local_conflicts:
                    best_local_conflicts = n_conflicts
                    best_local = [r[:] for r in rows]
            else:
                # No improving move found for this element; try a random perturbation
                # to escape local minimum
                if random.random() < 0.3:
                    # Pick a random element and move it randomly
                    ri2 = random.randint(0, n_rows - 1)
                    ei2 = random.randint(1, k)
                    row2 = rows[ri2]
                    lower2 = row2[ei2 - 1] + 1
                    upper2 = row2[ei2 + 1] - 1 if ei2 < k else max_scope
                    if lower2 <= upper2:
                        new_val2 = random.randint(lower2, upper2)
                        if new_val2 != row2[ei2]:
                            row2[ei2] = new_val2
                            diff_map = compute_all_diffs(rows)
                            n_conflicts = count_conflicts(diff_map)
                            elem_conflict_count = update_elem_conflicts(diff_map)
                            
                            if n_conflicts < best_local_conflicts:
                                best_local_conflicts = n_conflicts
                                best_local = [r[:] for r in rows]
            
            iter_count += 1
        
        return best_local, best_local_conflicts
    
    # Multi-start strategy:
    # 1. Start from greedy solution
    # 2. Try targeted backtracking with decreasing max_scope
    # 3. If stuck, perturb and retry
    
    current_rows = [r[:] for r in greedy_rows]
    current_scope = best_scope
    
    # Try to reduce scope
    for target_scope in range(best_scope - 1, max(M_lower - 1, best_scope - 30), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random starts around the target scope
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Start from a perturbed version of best_rows
            start_rows = [r[:] for r in best_rows]
            
            # Perturb: move some elements to random positions within target_scope
            num_perturb = random.randint(3, 8)
            for _ in range(num_perturb):
                ri = random.randint(0, n_rows - 1)
                ei = random.randint(1, k)
                row = start_rows[ri]
                lower = row[ei - 1] + 1
                upper = row[ei + 1] - 1 if ei < k else target_scope
                if lower <= upper:
                    row[ei] = random.randint(lower, upper)
            
            # Run targeted backtracking
            result_rows, result_conflicts = targeted_backtrack(
                start_rows, target_scope, deadline - 10
            )
            
            if result_conflicts == 0:
                # Valid solution with target_scope
                new_scope = max(max(r) for r in result_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in result_rows]
                break  # Found valid for this target, try smaller
    
    # If we haven't improved much, do a final SA pass to polish
    if time.time() < deadline - 15:
        # Simulated annealing polish on best_rows
        current_rows = [r[:] for r in best_rows]
        
        def compute_diff_counts_array(rows, max_val):
            counts = [0] * (max_val + 1)
            for row in rows:
                n = len(row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = row[j] - row[i]
                        if d <= max_val:
                            counts[d] += 1
            return counts
        
        max_val = best_scope + 50
        diff_counts = compute_diff_counts_array(current_rows, max_val)
        
        def get_num_dupes_from_counts(counts):
            return sum(1 for c in counts[1:] if c > 1)
        
        cur_dupes = get_num_dupes_from_counts(diff_counts)
        
        T = 5.0
        alpha = 0.998
        min_T = 0.01
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and time.time() < deadline - 5:
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            row = current_rows[ri]
            old_val = row[ei]
            
            lower = row[ei - 1] + 1
            upper = row[ei + 1] - 1 if ei < k else best_scope
            
            if lower > upper:
                steps += 1
                continue
            
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                steps += 1
                continue
            
            # Calculate delta
            delta = 0
            for j in range(len(row)):
                if j == ei:
                    continue
                d_old = abs(row[j] - old_val)
                if d_old <= max_val:
                    if diff_counts[d_old] == 2:
                        delta -= 1
                
                d_new = abs(row[j] - new_val)
                if d_new <= max_val:
                    if diff_counts[d_new] == 1:
                        delta += 1
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < math.exp(-delta * 2.0 / T)
            
            if accept:
                # Apply
                for j in range(len(row)):
                    if j == ei:
                        continue
                    d_old = abs(row[j] - old_val)
                    if d_old <= max_val:
                        diff_counts[d_old] -= 1
                    d_new = abs(row[j] - new_val)
                    if d_new <= max_val:
                        diff_counts[d_new] += 1
                row[ei] = new_val
                cur_dupes += delta
                
                if cur_dupes == 0:
                    new_scope = max(max(r) for r in current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
            
            T *= alpha
            if T < min_T:
                T = min_T
            steps += 1
    
    # Final validation
    used_diffs = set()
    valid = True
    for row in best_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used_diffs:
                    valid = False
                    break
                used_diffs.add(d)
        if not valid:
            break
    
    if valid:
        return best_rows
    else:
        return greedy_rows
