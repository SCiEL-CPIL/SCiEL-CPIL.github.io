import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_rows = None
    best_max = float('inf')
    
    # Initial greedy solution to set a good upper bound
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        greedy_rows.append(row)
    best_max = max(max(r) for r in greedy_rows)
    best_rows = greedy_rows
    
    # Bitset-based CSP with forward checking and randomized restarts
    # We use bitmasks for efficient difference checks.
    # Max possible diff is bounded by best_max initially, but we search for lower targets.
    # Let's define a maximum search bound. The problem size is small (k=8, n_rows=4).
    # Max entry ~134. Diffs are 1..133.
    # We can use an integer bitmask where bit d is set if diff d is used.
    
    def try_construct(target_max, n_rows, k, random_state):
        used_diff_mask = 0
        rows = []
        
        # Precompute valid candidates for each row position given current mask?
        # No, mask changes. But we can precompute which x values are valid for a given row prefix and mask?
        # That's too expensive.
        
        # Instead, we check on the fly.
        # To speed up, we can precompute for each x (1..target_max) the bitmask of diffs it introduces
        # relative to a row prefix? No, it depends on the row prefix.
        
        # Let's just use sets for clarity first, then optimize if needed.
        # Actually, bitmask operations are faster.
        # diff d is represented as bit (d-1) or just d?
        # Let's use bit d for diff d.
        
        def get_diff_mask_for_row(row):
            mask = 0
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
            return mask

        def backtrack_row(idx, row, used_diff_mask):
            if idx == k:
                return row[:]
            
            last = row[-1]
            
            # Collect valid candidates
            valid_candidates = []
            for x in range(last + 1, target_max + 1):
                # Check if adding x creates any conflict
                # Diffs are x - a for a in row
                # We need to check if any of these diffs are already in used_diff_mask
                # And they must be distinct (they are, since row is strictly increasing)
                
                conflict = False
                for a in row:
                    d = x - a
                    if (used_diff_mask >> d) & 1:
                        conflict = True
                        break
                
                if not conflict:
                    valid_candidates.append(x)
                    
            if not valid_candidates:
                return None
            
            # Heuristic: Prefer smaller x, but randomize
            # We can shuffle a subset of the smallest candidates
            W = 10
            candidates_to_try = valid_candidates[:W] if len(valid_candidates) > W else valid_candidates
            random_state.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                new_row = row + [x]
                # Calculate new diffs mask
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                
                res = backtrack_row(idx + 1, new_row, used_diff_mask | new_diffs_mask)
                if res is not None:
                    return res
                    
            return None

        # Main loop for rows
        for i in range(n_rows):
            row = [0]
            res = backtrack_row(0, row, used_diff_mask)
            if res is None:
                return None
            # Update global mask
            used_diff_mask |= get_diff_mask_for_row(res)
            rows.append(res)
            
        return rows

    # Main search loop
    # We want to minimize the maximum entry.
    # Start from best_max - 1 and go down.
    # If we find a solution, update best_max and continue.
    
    current_target = best_max - 1
    
    while time.time() < deadline and current_target >= 0:
        found = False
        # Try multiple restarts for this target
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            random_state = random.Random()
            res = try_construct(current_target, n_rows, k, random_state)
            if res is not None:
                current_max = max(max(r) for r in res)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = res
                found = True
                break
        
        if found:
            current_target = best_max - 1
        else:
            # If we can't find a solution for current_target, 
            # it might be impossible or just hard.
            # We can try to reduce the target further? No, that's harder.
            # We should probably stop if we fail for a target that is close to best_max?
            # Or we can just break if we've spent too much time on one target.
            # For now, if we fail, we assume it's hard and stop? 
            # Or we can try a different target?
            # Let's just break if we fail, as lower targets are harder.
            # Actually, if target_max is too small, it's impossible.
            # If it's just hard, more restarts might help.
            # Let's try a few more restarts if time permits.
            
            # Let's try to see if we can find ANY solution with a slightly larger max?
            # No, we want to minimize.
            
            # If we fail, we just keep the current best.
            break

    return best_rows
