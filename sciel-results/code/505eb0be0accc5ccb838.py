def solve_csp_random(target_scope, seed):
    random.seed(seed)
    used_mask = 0
    rows = []
    
    def backtrack_row(r, current_row, current_mask):
        nonlocal used_mask, rows
        if time.time() - scope_start > time_per_scope:
            return False
        
        if r == n_rows:
            return True
        
        if len(current_row) == k + 1:
            new_used_mask = used_mask | current_mask
            rows.append(current_row[:])
            if backtrack_row(r + 1, [0], 0):
                return True
            rows.pop()
            return False
        
        min_x = current_row[-1] + 1
        max_x = target_scope
        
        # Collect valid candidates
        candidates = []
        for x in range(min_x, max_x + 1):
            valid = True
            for a in current_row:
                d = x - a
                if (used_mask >> d) & 1 or (current_mask >> d) & 1:
                    valid = False
                    break
            if valid:
                candidates.append(x)
        
        if not candidates:
            return False
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        for x in candidates:
            # Re-check validity? No, we checked above.
            # But we need to update current_mask
            new_diffs_mask = 0
            for a in current_row:
                new_diffs_mask |= (1 << (x - a))
            
            current_row.append(x)
            if backtrack_row(r, current_row, current_mask | new_diffs_mask):
                return True
            current_row.pop()
        
        return False
    
    return backtrack_row(0, [0], 0)
