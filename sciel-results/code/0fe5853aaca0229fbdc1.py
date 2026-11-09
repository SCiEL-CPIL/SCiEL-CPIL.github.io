import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit
    
    # k+1 integers, starting at 0.
    # n_rows rows.
    
    def greedy_construct():
        used = 0  # Bitset of used differences
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try next integer
                x = row[-1] + 1
                while True:
                    # Check differences with existing row elements
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if d <= 0:
                            continue
                        # Check if d is already in used
                        if (used >> d) & 1:
                            valid = False
                            break
                        # Check if d is already in new_diffs (duplicate within new additions)
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)

    # Get initial bound
    greedy_rows, initial_max = greedy_construct()
    best_rows = [r[:] for r in greedy_rows]
    best_max = initial_max
    
    # Local Search to improve
    def get_used_mask(rows):
        mask = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
        return mask

    def try_local_search(rows, current_max, max_iter=10000):
        used_mask = get_used_mask(rows)
        rows = [r[:] for r in rows]
        best_local_max = current_max
        best_local_rows = [r[:] for r in rows]
        
        # We want to reduce the maximum element.
        # Strategy: Pick a random row, pick a random element index (1 to end),
        # try to replace it with a smaller valid value.
        
        for _ in range(max_iter):
            if time.time() - start_time > timeout:
                break
            
            # Pick a random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick a random element to move (index 1 to len-1)
            # Bias towards larger indices (larger values) to reduce max faster?
            # Or just random.
            e_idx = random.randint(1, len(row) - 1)
            old_val = row[e_idx]
            
            # If this is not the global max, we might still want to move it to free up space?
            # Actually, reducing any element might help.
            
            # Remove old_val from used_mask
            # We need to clear the bits for differences involving old_val in this row.
            # It's easier to just recompute the mask for this row and XOR?
            # No, just recompute the whole mask? Too slow.
            # Let's compute the mask for this row, and the mask for the rest.
            
            # Actually, let's just try to find a new value x < old_val.
            # We need to check if x is valid given the OTHER elements in the row and the GLOBAL used mask (excluding diffs from this row).
            
            # 1. Compute diffs of row excluding old_val.
            # Let row_without = row[:e_idx] + row[e_idx+1:]
            # This is not a valid row structure for diff calculation if we just remove, but we can calculate diffs between remaining elements.
            
            # Better:
            # 1. Identify all diffs involving old_val in this row.
            #    These are old_val - row[i] for i < e_idx and row[j] - old_val for j > e_idx.
            #    Let's call this set D_old.
            # 2. The "available" diffs are global_used_mask XOR (mask of D_old).
            #    Wait, other rows might have used these diffs? No, D_old are unique to this row's internal structure?
            #    No, the constraint is that ALL diffs in ALL rows are distinct.
            #    So if we remove old_val, the diffs D_old become "free".
            #    We can reuse them if they don't conflict with other diffs in the NEW configuration of this row.
            
            # So, available_mask = used_mask ^ (mask of D_old)
            # But we must ensure that the NEW diffs we introduce don't conflict with:
            # a) The remaining diffs in this row (which are still used)
            # b) The diffs in other rows (which are in used_mask and not in D_old)
            
            # So, valid_new_diffs must not be in (used_mask ^ D_old_mask) ?
            # No.
            # Current used_mask contains:
            # - Diffs from other rows.
            # - Diffs from this row (including D_old and D_rest).
            
            # If we replace old_val with new_val:
            # - D_old are removed.
            # - D_rest remain.
            # - D_new (involving new_val) are added.
            
            # Constraint:
            # 1. D_new must be disjoint from D_rest.
            # 2. D_new must be disjoint from (used_mask - D_old - D_rest). i.e., diffs from other rows.
            
            # So, forbidden_mask = (used_mask ^ D_old_mask) & ~(D_rest_mask)
            # Wait, used_mask = D_other | D_rest | D_old.
            # forbidden = D_other | D_rest.
            # So forbidden_mask = used_mask ^ D_old_mask.
            
            # So we just need D_new to be disjoint from (used_mask ^ D_old_mask).
            
            # Let's compute D_old_mask and D_rest_mask.
            D_old_mask = 0
            D_rest_mask = 0
            
            # Diffs involving old_val
            for i in range(len(row)):
                if i == e_idx: continue
                d = abs(row[i] - old_val)
                D_old_mask |= (1 << d)
                
            # Diffs NOT involving old_val
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    if i == e_idx or j == e_idx: continue
                    d = row[j] - row[i]
                    D_rest_mask |= (1 << d)
            
            forbidden_mask = used_mask ^ D_old_mask
            
            # Now try to find new_val < old_val such that:
            # 1. row is strictly increasing: row[e_idx-1] < new_val < row[e_idx+1] (if exist)
            # 2. Diffs involving new_val are not in forbidden_mask.
            
            lower_bound = row[e_idx-1] + 1 if e_idx > 0 else 1
            upper_bound = row[e_idx+1] - 1 if e_idx < len(row)-1 else old_val - 1
            
            if lower_bound > upper_bound:
                continue
                
            # Try values from upper_bound down to lower_bound?
            # To reduce max, we want the smallest possible change?
            # Actually, we just want ANY valid new_val < old_val.
            # Picking a smaller one is better for the max, but might be harder to find.
            # Let's try a random subset or just iterate?
            # Iterate is deterministic. Randomize start?
            
            # Let's try values from lower_bound to upper_bound.
            # If we find one, we replace.
            
            found = False
            # To add randomness, we can shuffle the range?
            # Or just pick a random candidate from the range?
            # Let's try a few random candidates first, then sequential?
            # For simplicity, let's just try sequential from lower_bound.
            # This might get stuck if the only valid small number is blocked.
            # But since we are freeing up D_old, we have more space.
            
            # Optimization: If we just want to reduce the MAX, we should focus on the max element.
            # But local search usually explores randomly.
            
            # Let's try to find a new_val.
            for new_val in range(lower_bound, upper_bound + 1):
                # Check diffs
                new_diffs_mask = 0
                valid = True
                for i in range(len(row)):
                    if i == e_idx: continue
                    d = abs(row[i] - new_val)
                    if (forbidden_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    # Accept move
                    # Update used_mask
                    used_mask = (used_mask ^ D_old_mask) | new_diffs_mask
                    # Update row
                    row[e_idx] = new_val
                    # Note: row must remain sorted.
                    # Since lower_bound > row[e_idx-1] and upper_bound < row[e_idx+1], it is sorted.
                    
                    # Update best if improved
                    current_row_max = max(row)
                    global_max = max(max(r) for r in rows)
                    if global_max < best_local_max:
                        best_local_max = global_max
                        best_local_rows = [r[:] for r in rows]
                    
                    found = True
                    break
            
            if not found:
                # Try a random value in range?
                pass # Keep it simple for now

        return best_local_rows, best_local_max

    # Run local search
    improved_rows, improved_max = try_local_search(best_rows, best_max, max_iter=50000)
    if improved_max < best_max:
        best_rows = improved_rows
        best_max = improved_max

    return best_rows
# EVOLVE-BLOCK-END
