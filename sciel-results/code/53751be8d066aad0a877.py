import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 120.0  # Keep a margin from the 150s limit
    
    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Initialize rows greedily
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            # Try next integer
            x = current_max + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    break
                x += 1
            row.append(x)
            current_max = x
            used_diffs |= new_diffs
        rows.append(row)
    
    # Check if greedy already found a valid solution (it does, but maybe suboptimal scope)
    # We will now try to improve the scope.
    
    def get_scope(rows):
        return max(max(r) for r in rows)
        
    def count_conflicts(rows, used_diffs):
        # This is expensive if done from scratch. 
        # We will maintain used_diffs and a conflict count dynamically in the loop.
        pass

    # State: rows, used_diffs (set), conflict_count (int), current_scope (int)
    # We need to track conflicts carefully.
    # A difference is a conflict if it appears more than once.
    # Let's maintain a dictionary: diff -> count.
    
    diff_counts = {}
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
    conflict_count = sum(1 for v in diff_counts.values() if v > 1)
    current_scope = get_scope(rows)
    
    # Cost function: We want 0 conflicts. Then minimize scope.
    # cost = conflict_count * 10000 + current_scope
    
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    best_conflicts = conflict_count
    
    T = 50.0
    cooling = 0.995
    
    while time.time() < deadline - 5.0 and T > 0.1:
        # Pick a random row and position
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k) # Position 1 to k (0 is fixed at 0)
        
        row = rows[r_idx]
        old_val = row[p_idx]
        
        # Determine range for new value
        # Must be > row[p_idx-1] and < row[p_idx+1] (if exists)
        min_val = row[p_idx-1] + 1
        max_val = row[p_idx+1] - 1 if p_idx < k else current_scope + 10 # Allow expansion? 
        # Actually, to reduce scope, we usually want to try smaller values.
        # But we might need to move out to resolve conflicts.
        # Let's try a random value in [min_val, max(current_scope, old_val + 10)]
        
        upper_bound = max(current_scope, old_val + 10)
        if min_val > upper_bound:
            continue
            
        new_val = random.randint(min_val, upper_bound)
        if new_val == old_val:
            continue
            
        # Calculate change in differences
        # Remove old differences involving p_idx
        removed = []
        for i in range(len(row)):
            if i != p_idx:
                d = abs(row[i] - old_val)
                removed.append(d)
                
        # Add new differences involving p_idx
        added = []
        for i in range(len(row)):
            if i != p_idx:
                d = abs(row[i] - new_val)
                added.append(d)
                
        # Update diff_counts and conflict_count
        # This is a bit complex to do exactly without applying the move, 
        # but we can simulate it.
        
        # Helper to apply/unapply diff changes
        def apply_diff_changes(diffs, sign, counts, conflicts):
            # sign: 1 for add, -1 for remove
            # Returns new conflict count
            c = conflicts
            for d in diffs:
                old_c = counts.get(d, 0)
                new_c = old_c + sign
                if new_c == 0:
                    del counts[d]
                else:
                    counts[d] = new_c
                    
                # Conflict logic:
                # A diff is a conflict if count > 1.
                # If old_c == 1 and new_c == 2: conflict increases by 1
                # If old_c == 2 and new_c == 1: conflict decreases by 1
                # If old_c > 2 and new_c > 1: no change in conflict count (still >1)
                # If old_c == 1 and new_c == 0: no change
                # If old_c == 0 and new_c == 1: no change
                
                if old_c == 1 and new_c == 2:
                    c += 1
                elif old_c == 2 and new_c == 1:
                    c -= 1
            return c

        # Simulate removal
        temp_counts = dict(diff_counts) # Copying dict is O(N), N is #diffs ~ 100-200. Acceptable.
        temp_conflicts = conflict_count
        temp_conflicts = apply_diff_changes(removed, -1, temp_counts, temp_conflicts)
        temp_conflicts = apply_diff_changes(added, 1, temp_counts, temp_conflicts)
        
        # Check if move is valid regarding row order (already ensured by min/max)
        
        new_scope = max(current_scope, new_val)
        # Note: if we remove the only max, scope might decrease. 
        # But since we pick random new_val, it's hard to know if it was the unique max.
        # For simplicity, let's just use the current_scope for cost if new_val < current_scope,
        # and new_val if new_val > current_scope.
        # Actually, we should calculate the true new scope.
        # max(row without p_idx) vs new_val.
        # This is O(k).
        rest_max = 0
        for i in range(len(row)):
            if i != p_idx:
                if row[i] > rest_max:
                    rest_max = row[i]
        new_scope = max(rest_max, new_val)
        
        # Check global scope
        # If this row's max decreases, global scope might decrease.
        # If this row's max increases, global scope might increase.
        # Let's just use the max of all rows' maxes.
        # To be safe and fast, let's approximate or just accept the move and recalc if accepted?
        # No, we need cost for acceptance.
        # Let's compute new global scope.
        new_global_scope = 0
        for idx, r in enumerate(rows):
            if idx == r_idx:
                m = new_scope
            else:
                m = max(r)
            if m > new_global_scope:
                new_global_scope = m
                
        old_cost = conflict_count * 10000 + current_scope
        new_cost = temp_conflicts * 10000 + new_global_scope
        
        if new_cost < old_cost or random.random() < math.exp((old_cost - new_cost) / T):
            # Accept move
            row[p_idx] = new_val
            diff_counts = temp_counts
            conflict_count = temp_conflicts
            current_scope = new_global_scope
            
            if conflict_count == 0:
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_conflicts = 0
                    best_rows = [r[:] for r in rows]
                    
        T *= cooling
        
    # Return the best valid solution found
    # If best_conflicts > 0, we must return a valid solution.
    # The greedy solution is always valid.
    if best_conflicts > 0:
        return [r[:] for r in rows] # Wait, rows might be invalid now if we accepted invalid moves?
        # No, we only accept moves based on cost. If conflict_count > 0, cost is high.
        # But we might have ended in a state with conflicts.
        # We must ensure the returned solution is VALID.
        # The initial greedy solution is valid.
        # Let's keep the initial greedy solution as a fallback.
        
    # Re-evaluate best_rows validity just in case
    # Actually, if we tracked conflicts correctly, best_conflicts == 0 implies valid.
    # But if we never found a valid solution better than greedy, best_rows is the greedy one?
    # No, best_rows is only updated if conflict_count == 0.
    # If we never improved, best_rows is the initial greedy one (which is valid).
    
    return best_rows
