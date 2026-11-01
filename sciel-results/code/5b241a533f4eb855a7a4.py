import time
import random
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit (with 8s buffer)
    
    # Each row has k+1 elements (indices 0..k), starting at 0
    # Total differences per row: (k+1)*k/2
    # Total differences across all rows: n_rows * (k+1)*k/2
    
    total_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    # Counting bound: need at least total_diffs_needed distinct differences
    # The largest element M must satisfy: number of distinct positive integers <= M is M
    # But differences can be up to M, and we need total_diffs_needed distinct ones
    # So M >= total_diffs_needed (roughly)
    # More precisely, differences range from 1 to M, so we need M >= total_diffs_needed
    counting_bound = total_diffs_needed
    
    def get_diffs_for_row(row):
        """Get all pairwise differences for a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def count_all_conflicts(rows):
        """Count total number of duplicate differences across all rows."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_diffs_for_row(row))
        c = Counter(all_diffs)
        conflicts = sum(cnt - 1 for cnt in c.values() if cnt > 1)
        return conflicts
    
    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(r) for r in rows)
    
    def greedy_init(target_max=None):
        """
        Greedily construct rows, picking smallest available values.
        If target_max is given, limit search to that range.
        """
        if target_max is None:
            target_max = 500  # generous upper bound
        
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                for x in range(current_max + 1, target_max + 1):
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
                        found = True
                        break
                if not found:
                    # Failed to complete this row within target_max
                    return None
            rows.append(row)
        return rows
    
    def local_search(rows, max_iterations=5000):
        """
        Local search to reduce the scope.
        We try to reduce the largest element by reassigning values.
        """
        best_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_conflicts = count_all_conflicts(rows)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            if best_conflicts == 0:
                # Valid solution, try to reduce scope
                # Find the row with the largest element
                max_row_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
                max_val = max(best_rows[max_row_idx])
                
                if max_val <= counting_bound:
                    break  # Can't do better than counting bound
                
                # Try to reduce elements in the row with max value
                # Pick a random element position (not 0)
                pos = random.randint(1, k)
                row = best_rows[max_row_idx]
                
                if pos >= len(row):
                    continue
                    
                old_val = row[pos]
                
                # Try smaller values
                lower = row[pos-1] + 1 if pos > 0 else 1
                upper = old_val - 1
                
                if lower > upper:
                    continue
                
                # Try a few random smaller values
                candidates = []
                for _ in range(20):
                    v = random.randint(lower, min(upper, old_val - 1))
                    candidates.append(v)
                candidates.append(upper)  # Also try the immediate smaller value
                
                random.shuffle(candidates)
                
                improved = False
                for new_val in candidates:
                    if new_val >= old_val:
                        continue
                    
                    # Apply the change
                    row[pos] = new_val
                    
                    # Check if row is still strictly increasing
                    valid_struct = True
                    for i in range(1, len(row)):
                        if row[i] <= row[i-1]:
                            valid_struct = False
                            break
                    if not valid_struct:
                        row[pos] = old_val
                        continue
                    
                    # Check if this creates conflicts
                    new_conflicts = count_all_conflicts(best_rows)
                    new_scope = get_scope(best_rows)
                    
                    if new_conflicts == best_conflicts and new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in best_rows]
                        improved = True
                        break
                    else:
                        row[pos] = old_val
                
                if improved:
                    continue
            
            # If we have conflicts, try to resolve them
            if best_conflicts > 0:
                # Find a conflicting difference
                all_diffs = []
                for row in best_rows:
                    all_diffs.extend(get_diffs_for_row(row))
                c = Counter(all_diffs)
                conflicting_diffs = [d for d, cnt in c.items() if cnt > 1]
                
                if not conflicting_diffs:
                    continue
                
                target_diff = random.choice(conflicting_diffs)
                
                # Find rows that have this difference
                rows_with_diff = []
                for i, row in enumerate(best_rows):
                    if target_diff in get_diffs_for_row(row):
                        rows_with_diff.append(i)
                
                if len(rows_with_diff) < 2:
                    continue
                
                # Pick one of the rows and try to modify it
                mod_row_idx = random.choice(rows_with_diff)
                row = best_rows[mod_row_idx]
                
                # Try modifying a random position
                pos = random.randint(1, k)
                if pos >= len(row):
                    continue
                    
                old_val = row[pos]
                lower = row[pos-1] + 1 if pos > 0 else 1
                upper = row[pos+1] - 1 if pos < k else old_val + 50
                
                if lower > upper:
                    continue
                
                # Try random values in range
                for _ in range(30):
                    new_val = random.randint(lower, upper)
                    if new_val == old_val:
                        continue
                    
                    row[pos] = new_val
                    
                    # Check structure
                    valid_struct = True
                    for i in range(1, len(row)):
                        if row[i] <= row[i-1]:
                            valid_struct = False
                            break
                    if not valid_struct:
                        row[pos] = old_val
                        continue
                    
                    new_conflicts = count_all_conflicts(best_rows)
                    new_scope = get_scope(best_rows)
                    
                    if new_conflicts < best_conflicts:
                        best_conflicts = new_conflicts
                        best_rows = [r[:] for r in best_rows]
                        break
                    elif new_conflicts == best_conflicts and new_scope < best_scope:
                        best_conflicts = new_conflicts
                        best_scope = new_scope
                        best_rows = [r[:] for r in best_rows]
                        break
                    else:
                        row[pos] = old_val
        
        return best_rows
    
    def construct_best_row_with_conflicts(used_diffs_from_others, target_max=None):
        """
        Construct a single row that avoids conflicts with used_diffs_from_others.
        Returns the row or None.
        """
        if target_max is None:
            target_max = 300
        
        row = [0]
        current_max = 0
        current_diffs = set()
        
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, target_max + 1):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs_from_others or d in current_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    current_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    found = True
                    break
            
            if not found:
                return None
        
        return row
    
    # Main algorithm:
    # 1. Start with greedy initialization
    # 2. Try to reduce scope via local search
    # 3. Use row-replacement strategy to improve
    
    # Step 1: Greedy initialization
    initial_rows = greedy_init()
    if initial_rows is None:
        # Fallback: even more generous
        initial_rows = greedy_init(target_max=1000)
    
    if initial_rows is None:
        # Last resort: just return something (shouldn't happen)
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
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
    
    best_rows = [r[:] for r in initial_rows]
    best_scope = get_scope(best_rows)
    best_conflicts = count_all_conflicts(best_rows)
    
    # Step 2: Local search to improve
    best_rows = local_search(best_rows, max_iterations=10000)
    best_scope = get_scope(best_rows)
    best_conflicts = count_all_conflicts(best_rows)
    
    # Step 3: Iterative row replacement
    # Try to replace individual rows with better-fitting ones
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        
        if best_conflicts == 0:
            # Valid solution, try to reduce scope further
            # Find the row with the largest max
            max_row_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
            
            # Compute used diffs from other rows
            used_diffs_from_others = set()
            for j in range(n_rows):
                if j == max_row_idx:
                    continue
                used_diffs_from_others.update(get_diffs_for_row(best_rows[j]))
            
            # Try to construct a better row
            current_max = max(best_rows[max_row_idx])
            
            # Try target_max values from current_max-1 down to counting_bound
            found_better = False
            for tm in range(current_max - 1, counting_bound - 1, -1):
                if time.time() > deadline - 5:
                    break
                
                new_row = construct_best_row_with_conflicts(used_diffs_from_others, target_max=tm)
                if new_row is not None:
                    # Check if this improves scope
                    new_scope_candidate = max(
                        max(best_rows[max_row_idx] if i != max_row_idx else new_row) 
                        for i in range(n_rows)
                    )
                    if new_scope_candidate < best_scope:
                        best_rows[max_row_idx] = new_row
                        best_scope = new_scope_candidate
                        found_better = True
                        break
            
            if found_better:
                # Continue local search after improvement
                best_rows = local_search(best_rows, max_iterations=2000)
                best_scope = get_scope(best_rows)
                best_conflicts = count_all_conflicts(best_rows)
            else:
                # Can't improve via row replacement, try local search again
                improved_rows = local_search(best_rows, max_iterations=5000)
                new_scope = get_scope(improved_rows)
                new_conflicts = count_all_conflicts(improved_rows)
                if new_scope < best_scope or (new_scope == best_scope and new_conflicts < best_conflicts):
                    best_rows = improved_rows
                    best_scope = new_scope
                    best_conflicts = new_conflicts
        
        else:
            # Have conflicts, try to resolve by replacing a problematic row
            all_diffs = []
            for row in best_rows:
                all_diffs.extend(get_diffs_for_row(row))
            c = Counter(all_diffs)
            conflicting_diffs = [d for d, cnt in c.items() if cnt > 1]
            
            if conflicting_diffs:
                # Find a row that's involved in many conflicts
                row_conflict_counts = [0] * n_rows
                for i, row in enumerate(best_rows):
                    row_diffs = get_diffs_for_row(row)
                    for d in row_diffs:
                        if d in conflicting_diffs:
                            row_conflict_counts[i] += 1
                
                # Pick the row with most conflicts
                worst_row = max(range(n_rows), key=lambda i: row_conflict_counts[i])
                
                # Compute used diffs from other rows
                used_diffs_from_others = set()
                for j in range(n_rows):
                    if j == worst_row:
                        continue
                    used_diffs_from_others.update(get_diffs_for_row(best_rows[j]))
                
                # Try to construct a new row
                current_max = max(best_rows[worst_row])
                new_row = construct_best_row_with_conflicts(used_diffs_from_others, target_max=current_max + 50)
                
                if new_row is not None:
                    best_rows[worst_row] = new_row
                    best_conflicts = count_all_conflicts(best_rows)
                    best_scope = get_scope(best_rows)
                    continue
            
            # Try local search
            improved_rows = local_search(best_rows, max_iterations=5000)
            new_scope = get_scope(improved_rows)
            new_conflicts = count_all_conflicts(improved_rows)
            if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_scope < best_scope):
                best_rows = improved_rows
                best_scope = new_scope
                best_conflicts = new_conflicts
    
    return best_rows
