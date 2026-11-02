import time

def construct(n_rows, k):
    start_time = time.time()
    # 10s margin from 120s limit
    deadline = start_time + 110.0
    
    # Lower bound for the maximum element M
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Helper to generate all valid rows (Golomb rulers) up to max_val
    def generate_rows(max_val):
        rows = []
        def backtrack(current_row, current_diffs, next_idx):
            if time.time() > deadline:
                return
            if next_idx == k + 1:
                rows.append((list(current_row), frozenset(current_diffs)))
                return
            
            start_val = current_row[-1] + 1 if current_row else 1
            # We need to fit (k + 1 - next_idx - 1) more numbers.
            # Min step is 1.
            # So current_val + (remaining_count) <= max_val
            remaining_count = k - next_idx
            end_val = min(max_val, max_val - remaining_count)
            
            for v in range(start_val, end_val + 1):
                new_diffs = set()
                valid = True
                for u in current_row:
                    d = v - u
                    if d in current_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    current_row.append(v)
                    current_diffs.update(new_diffs)
                    backtrack(current_row, current_diffs, next_idx + 1)
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
        
        backtrack([0], set(), 1)
        return rows

    def select_rows(candidates, n_needed, used_diffs, current_selection):
        if time.time() > deadline:
            return None
        
        if n_needed == 0:
            return current_selection
        
        if len(candidates) < n_needed:
            return None
            
        # If we need exactly as many as available, check if they are all compatible
        if len(candidates) == n_needed:
            # Check if all pairs are disjoint
            # Since we check against used_diffs recursively, we just need to ensure
            # that adding all of them doesn't cause conflicts.
            # Actually, the recursion handles this. But we can shortcut.
            # Let's just let the recursion handle it, it's fast enough.
            pass

        for i, (row, diffs) in enumerate(candidates):
            if not diffs & used_diffs:
                result = select_rows(candidates[i+1:], n_needed - 1, used_diffs | diffs, current_selection + [row])
                if result is not None:
                    return result
        return None

    # Iterate M from lower bound
    # Search up to lower_bound + 50
    for M in range(lower_bound, lower_bound + 50):
        if time.time() > deadline:
            break
        
        candidates = generate_rows(M)
        
        if len(candidates) < n_rows:
            continue
            
        solution = select_rows(candidates, n_rows, frozenset(), [])
        
        if solution:
            return solution
            
    # Fallback: Greedy construction
    used, rows = set(), []
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
