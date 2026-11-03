import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # Greedy construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_row_diffs
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        row.append(x)
                        current_row_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
            used |= current_row_diffs
        return rows

    # Bitset-accelerated row generation
    def generate_row_bitset(used_diffs, k_val, max_scope, randomize=False, p_smallest=0.7, window=5):
        """Generate a single row using bitset for fast conflict checking."""
        # Create a bitset of forbidden differences
        max_d = max_scope + 1
        # Use a Python int as bitset: bit d is set if difference d is forbidden
        forbidden_bits = 0
        for d in used_diffs:
            if d < max_d:
                forbidden_bits |= (1 << d)
        
        row = [0]
        row_diffs_bits = 0  # Bitset of differences within current row
        
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            
            while x < max_scope:
                # Check if x creates any forbidden differences
                conflict = False
                for a in row:
                    d = x - a
                    if d < max_d and (forbidden_bits >> d) & 1:
                        conflict = True
                        break
                    if d < max_d and (row_diffs_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
            
            # Update row_diffs_bits
            for a in row:
                d = next_x - a
                if d < max_d:
                    row_diffs_bits |= (1 << d)
            
            row.append(next_x)
        
        return row

    # Branch and bound for maximum independent set on a small graph
    # Returns a set of positions (indices into 0..max_scope-1)
    def max_independent_set_branch_and_bound(conflict_matrix, n_positions, target_size):
        """
        Find a maximum independent set of size target_size in a graph.
        conflict_matrix: list of sets, where conflict_matrix[i] is the set of j that conflict with i.
        Returns a list of positions, or None if no independent set of size target_size exists.
        """
        if n_positions <= 0:
            return [] if target_size == 0 else None
        
        # Use a simple branch and bound
        # Order vertices by degree (heuristic: pick vertex with fewest conflicts first)
        degrees = [len(conflict_matrix[i]) for i in range(n_positions)]
        order = list(range(n_positions))
        order.sort(key=lambda i: degrees[i])
        
        # Map from original index to position in order
        pos_in_order = [0] * n_positions
        for pos, idx in enumerate(order):
            pos_in_order[idx] = pos
        
        # Try to build an independent set
        def backtrack(start_pos, current_set, remaining_needed):
            if remaining_needed == 0:
                return current_set[:]
            if start_pos >= len(order):
                return None
            
            # Upper bound: can we still reach target_size?
            if remaining_needed > len(order) - start_pos:
                return None
            
            # Try including order[start_pos]
            v = order[start_pos]
            # Check if v conflicts with any in current_set
            can_include = True
            for u in current_set:
                if u in conflict_matrix[v]:
                    can_include = False
                    break
            
            if can_include:
                result = backtrack(start_pos + 1, current_set + [v], remaining_needed - 1)
                if result is not None:
                    return result
            
            # Try excluding order[start_pos]
            result = backtrack(start_pos + 1, current_set, remaining_needed)
            if result is not None:
                return result
            
            return None
        
        result = backtrack(0, [], target_size)
        if result is not None:
            return sorted(result)
        return None

    # Local compaction: try to move each element to a smaller position
    def local_compact(rows, used_diffs, max_iterations=1000):
        """Try to reduce the scope by moving elements to smaller positions."""
        current_scope = max(max(r) for r in rows)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for p_idx in range(len(row) - 1, 0, -1):  # Start from last element
                    old_val = row[p_idx]
                    if old_val <= 1:
                        continue
                    
                    # Remove old_val from row
                    row.pop(p_idx)
                    
                    # Remove its differences from used_diffs
                    diffs_to_remove = set()
                    for a in row:
                        d = abs(old_val - a)
                        diffs_to_remove.add(d)
                    used_diffs -= diffs_to_remove
                    
                    # Try to find a smaller position
                    found = False
                    for x in range(old_val - 1, 0, -1):
                        if x in row:
                            continue
                        valid = True
                        for a in row:
                            d = abs(x - a)
                            if d == 0 or d in used_diffs:
                                valid = False
                                break
                        if valid:
                            # Insert x in sorted position
                            pos = 0
                            while pos < len(row) and row[pos] < x:
                                pos += 1
                            row.insert(pos, x)
                            for a in row:
                                if a != x:
                                    used_diffs.add(abs(x - a))
                            found = True
                            improved = True
                            break
                    
                    if not found:
                        # Re-insert old_val
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for a in row:
                            if a != old_val:
                                used_diffs.add(abs(old_val - a))
        
        return rows, used_diffs

    # 1. Initial Solution: Deterministic Greedy
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # 2. Simulated Annealing with Row Regeneration
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    used = get_used_diffs(rows)
    
    # Use more of the time budget for SA
    sa_deadline = deadline - 15
    while time.time() < sa_deadline:
        if T < T_min:
            T = 5.0
            cooling = 0.998
            
        r_idx = random.randint(0, n_rows - 1)
        
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        # Regenerate row with randomization
        new_row = generate_row_bitset(used, k, best_max + 20, randomize=True, p_smallest=0.8, window=3)
        
        if new_row is None:
            used |= old_diffs
            continue
        
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i + 1, len(new_row)))
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            used |= old_diffs
            
        T *= cooling

    # 3. Final Local Search: Individual Element Moves (optimized)
    current_rows = [list(r) for r in best_rows]
    
    diff_counts = {}
    current_conflict_count = 0
    
    def add_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            if diff_counts[d] == 1:
                current_conflict_count += 1
            diff_counts[d] += 1
        else:
            diff_counts[d] = 1
            
    def remove_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            elif diff_counts[d] == 1:
                current_conflict_count -= 1

    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                add_diff(d)
                
    current_scope = max(max(r) for r in current_rows)
    current_cost = current_scope + 1000 * current_conflict_count
    best_cost = current_cost
    best_rows_final = [list(r) for r in current_rows]
    best_scope_final = current_scope
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    steps = 0
    ls_deadline = deadline - 5
    while time.time() < ls_deadline:
        if T < T_min:
            T = 2.0
            cooling = 0.9995
            if steps > 80000:
                break
                
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, len(row) - 1)
        
        old_val = row[p_idx]
        
        # Build set of current row values for quick lookup
        row_set = set(row)
        
        diffs_to_remove = []
        for i in range(len(row)):
            if i != p_idx:
                diffs_to_remove.append(abs(row[i] - old_val))
        
        for d in diffs_to_remove:
            remove_diff(d)
        
        row.pop(p_idx)
        row_set.discard(old_val)
        
        new_val = None
        new_r_idx = r_idx
        
        if random.random() < 0.7:
            # Try to find a better (smaller) position in the same row
            max_search = min(best_scope_final + 5, 200)
            found = False
            # Try sequential smaller positions first for compaction
            for x in range(1, old_val):
                if x in row_set:
                    continue
                valid = True
                new_diffs = []
                for a in row:
                    d = abs(x - a)
                    if d in diff_counts:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < x:
                        pos += 1
                    row.insert(pos, x)
                    for d in new_diffs:
                        add_diff(d)
                    new_val = x
                    found = True
                    break
            
            if not found:
                # Try random positions
                for _ in range(5):
                    cand = random.randint(1, max_search)
                    if cand in row_set:
                        continue
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = abs(cand - a)
                        if d in diff_counts:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        pos = 0
                        while pos < len(row) and row[pos] < cand:
                            pos += 1
                        row.insert(pos, cand)
                        for d in new_diffs:
                            add_diff(d)
                        new_val = cand
                        found = True
                        break
            
            if not found:
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                row_set.add(old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        else:
            # Move element to another row
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            
            cand = old_val
            target_row = current_rows[new_r_idx]
            target_set = set(target_row)
            
            if cand in target_set:
                cand = random.randint(1, best_scope_final + 5)
                if cand in target_set:
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    row_set.add(old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue
            
            valid = True
            new_diffs = []
            for a in target_row:
                d = abs(cand - a)
                if d in diff_counts:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                pos = 0
                while pos < len(target_row) and target_row[pos] < cand:
                    pos += 1
                target_row.insert(pos, cand)
                for d in new_diffs:
                    add_diff(d)
                new_val = cand
            else:
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                row_set.add(old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        
        new_scope = max(max(r) for r in current_rows)
        new_cost = new_scope + 1000 * current_conflict_count
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            current_scope = new_scope
            
            if is_valid(current_rows):
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope_final = current_scope
                    best_rows_final = [list(r) for r in current_rows]
        else:
            # Revert
            if new_r_idx != r_idx:
                target_row = current_rows[new_r_idx]
                idx_new = target_row.index(new_val)
                target_row.pop(idx_new)
                for a in target_row:
                    remove_diff(abs(a - new_val))
            else:
                row = current_rows[r_idx]
                idx_new = row.index(new_val)
                row.pop(idx_new)
                for a in row:
                    remove_diff(abs(a - new_val))
            
            row = current_rows[r_idx]
            pos = 0
            while pos < len(row) and row[pos] < old_val:
                pos += 1
            row.insert(pos, old_val)
            for d in diffs_to_remove:
                add_diff(d)
        
        T *= cooling
        steps += 1

    # 4. Final Compaction: aggressive multi-pass element minimization
    final_rows = [list(r) for r in best_rows_final]
    used_set = get_used_diffs(final_rows)
    final_scope = max(max(r) for r in final_rows)
    
    # Multi-pass: try to move every element to its minimum valid position
    # Repeat until no improvement or time runs out
    for _pass in range(200):
        if time.time() > deadline - 1:
            break
        any_improved = False
        for row_idx in range(n_rows):
            row = final_rows[row_idx]
            # Process elements from last to first (higher elements have more room to move down)
            for p_idx in range(len(row) - 1, 0, -1):
                old_val = row[p_idx]
                if old_val <= 1:
                    continue
                
                # Remove old_val
                row.pop(p_idx)
                for a in row:
                    used_set.discard(abs(old_val - a))
                
                # Find minimum valid position
                min_pos = None
                row_set = set(row)
                for x in range(1, old_val):
                    if x in row_set:
                        continue
                    valid = True
                    for a in row:
                        d = abs(x - a)
                        if d == 0 or d in used_set:
                            valid = False
                            break
                    if valid:
                        min_pos = x
                        break
                
                if min_pos is not None and min_pos < old_val:
                    pos = 0
                    while pos < len(row) and row[pos] < min_pos:
                        pos += 1
                    row.insert(pos, min_pos)
                    for a in row:
                        if a != min_pos:
                            used_set.add(abs(min_pos - a))
                    any_improved = True
                else:
                    # Put back
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for a in row:
                        if a != old_val:
                            used_set.add(abs(old_val - a))
        
        if not any_improved:
            break
    
    final_scope = max(max(r) for r in final_rows)
    
    if is_valid(final_rows) and final_scope < best_scope_final:
        best_rows_final = final_rows
        best_scope_final = final_scope
        
    if not is_valid(best_rows_final):
        if is_valid(best_rows):
            best_rows_final = best_rows
        else:
            used = set()
            best_rows_final = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used:
                                valid = False
                                break
                        if valid:
                            break
                    row.append(x)
                    used.update({x - a for a in row if a != x})
                best_rows_final.append(row)

    return best_rows_final
