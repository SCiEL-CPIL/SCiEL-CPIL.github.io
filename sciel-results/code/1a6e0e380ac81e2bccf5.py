import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Greedy Solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_construct()
    
    # Helper to calculate cost and diff counts
    # diff_counts: dict mapping diff value to count
    # collision_count: number of diff values with count > 1
    # max_val: max entry in all rows
    
    def get_initial_stats(rows):
        diff_counts = {}
        max_val = 0
        for row in rows:
            for a in row:
                if a > max_val:
                    max_val = a
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collision_count = sum(1 for c in diff_counts.values() if c > 1)
        return diff_counts, collision_count, max_val

    diff_counts, collision_count, max_val = get_initial_stats(rows)
    best_rows = [r[:] for r in rows]
    best_cost = collision_count * 1000000 + max_val
    current_cost = best_cost
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute indices for moves
    # Move: pick row r, index i, delta
    # We only move indices 1..k (index 0 is fixed at 0)
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick an index to move. Index 0 is 0.
        # We can move indices 1 to k.
        # Moving index i affects diffs with all other elements.
        i = random.randint(1, k)
        
        # Pick a delta
        # Range of delta should be reasonable
        # Max possible value is roughly best_val
        # Let's try small deltas first, occasionally larger
        if random.random() < 0.8:
            delta = random.randint(-3, 3)
        else:
            delta = random.randint(-10, 10)
        
        new_val = row[i] + delta
        
        # Check increasing constraint
        if i > 0 and new_val <= row[i-1]:
            continue
        if i < k and new_val >= row[i+1]:
            continue
        
        # Calculate cost change
        # Remove old diffs involving row[i]
        old_val = row[i]
        removed_diffs = []
        for j in range(len(row)):
            if j == i: continue
            if j < i:
                d = old_val - row[j]
            else:
                d = row[j] - old_val
            removed_diffs.append(d)
            
        # Add new diffs involving new_val
        added_diffs = []
        for j in range(len(row)):
            if j == i: continue
            if j < i:
                d = new_val - row[j]
            else:
                d = row[j] - new_val
            added_diffs.append(d)
        
        # Update diff_counts temporarily to count collisions change
        # This is O(k) per move, which is fine.
        
        # We need to be careful: if a diff is in both removed and added, it cancels out.
        # Let's use a counter for the change in collision count.
        
        # To be efficient, we can just re-evaluate the affected diffs.
        # But maintaining global diff_counts is better.
        
        # Let's simulate the change:
        # For each d in removed_diffs:
        #   diff_counts[d] -= 1
        #   if diff_counts[d] was 2, collision decreases by 1 (2->1)
        #   if diff_counts[d] was 1, collision unchanged (1->0)
        #   if diff_counts[d] was >2, collision unchanged
        # For each d in added_diffs:
        #   diff_counts[d] += 1
        #   if diff_counts[d] was 1, collision increases by 1 (1->2)
        #   if diff_counts[d] was 0, collision unchanged (0->1)
        #   if diff_counts[d] was >=2, collision unchanged
        
        # We need to track the current diff_counts state.
        # Since we are doing SA, we might reject the move, so we must revert.
        
        # Optimization: Only calculate if it looks promising?
        # No, just calculate.
        
        delta_collisions = 0
        # Store original counts for reverted keys
        affected_keys = set(removed_diffs) | set(added_diffs)
        original_counts = {d: diff_counts.get(d, 0) for d in affected_keys}
        
        # Apply removals
        for d in removed_diffs:
            c = diff_counts.get(d, 0)
            if c == 0: 
                # Should not happen if state is consistent
                continue
            if c == 1:
                # 1 -> 0
                pass
            elif c == 2:
                # 2 -> 1
                delta_collisions -= 1
            else:
                # >2 -> >=2
                pass
            diff_counts[d] = c - 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Apply additions
        for d in added_diffs:
            c = diff_counts.get(d, 0)
            if c == 0:
                # 0 -> 1
                pass
            elif c == 1:
                # 1 -> 2
                delta_collisions += 1
            else:
                # >=2 -> >=3
                pass
            diff_counts[d] = c + 1
            
        # Calculate new max_val
        # We only changed row[r_idx][i]
        # If new_val > max_val, max_val increases.
        # If old_val == max_val and new_val < old_val, max_val might decrease if no other row has max_val.
        # To be safe and fast, let's just re-calculate max_val for the row, and check global.
        # Actually, tracking global max_val precisely during moves is tricky if multiple rows have the max.
        # Let's maintain a multiset or just re-calculate max_val from rows if collision count is 0?
        # Or simpler: just use the current max_val and update it heuristically?
        # No, let's just calculate the max of the modified row and compare with a stored global max excluding this row?
        # That's complex.
        
        # Alternative: Cost function is primarily collisions.
        # If collisions > 0, we don't care about max_val as much.
        # If collisions == 0, we care about max_val.
        
        # Let's just re-calculate max_val for the entire structure if the move is accepted?
        # That's O(n*k). Acceptable for n=4, k=8.
        
        # But we need it for the cost comparison BEFORE acceptance to decide.
        # So we need an efficient max_val update.
        
        # Let's maintain a list of max values per row.
        # row_maxs = [max(r) for r in rows]
        # global_max = max(row_maxs)
        
        # If we update row[r_idx], we update row_maxs[r_idx].
        # Then global_max = max(row_maxs).
        
        # Let's add this to the state.
        
        # Re-initializing state with row_maxs
        # This requires refactoring the loop slightly.
        
        # Let's do a quick check: if delta_collisions is large positive, reject immediately?
        # No, SA accepts with probability.
        
        # Let's implement the row_maxs tracking.
        
        # We need to initialize row_maxs.
        
        # I will rewrite the loop structure to include row_maxs.
        
        pass # Placeholder, will write full code below

    # The above sketch is getting complex to manage in a single thought block.
    # Let's write the actual code carefully.

    # Refactored Plan:
    # 1. Greedy Init
    # 2. State: rows, diff_counts, row_maxs, global_max, collision_count
    # 3. Loop:
    #    - Pick move
    #    - Check bounds
    #    - Calculate delta_collisions, new_row_max, new_global_max
    #    - Calculate delta_cost
    #    - Accept/Reject
    #    - If accepted, update state.
    #    - If rejected, revert state.

    # Let's implement this.

    row_maxs = [max(r) for r in rows]
    global_max = max(row_maxs)
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        if len(row) < 2:
            continue
            
        # Pick index 1..k
        i = random.randint(1, k)
        
        # Pick delta
        # Adaptive delta based on T?
        # Large T -> large moves.
        max_delta = int(T * 5) + 5
        if max_delta < 5: max_delta = 5
        
        if random.random() < 0.7:
            delta = random.randint(-max_delta, max_delta)
        else:
            delta = random.randint(-max_delta*2, max_delta*2)
            
        new_val = row[i] + delta
        
        # Check increasing
        if i > 0 and new_val <= row[i-1]:
            continue
        if i < k and new_val >= row[i+1]:
            continue
        
        old_val = row[i]
        
        # Calculate diff changes
        removed_diffs = []
        added_diffs = []
        
        for j in range(len(row)):
            if j == i: continue
            if j < i:
                removed_diffs.append(old_val - row[j])
                added_diffs.append(new_val - row[j])
            else:
                removed_diffs.append(row[j] - old_val)
                added_diffs.append(row[j] - new_val)
        
        # Calculate collision change
        delta_collisions = 0
        affected_keys = set(removed_diffs) | set(added_diffs)
        
        # We need to revert if rejected.
        # Store original counts for affected keys
        orig_counts = {}
        for d in affected_keys:
            orig_counts[d] = diff_counts.get(d, 0)
            
        # Apply removals
        for d in removed_diffs:
            c = diff_counts.get(d, 0)
            if c == 2:
                delta_collisions -= 1
            diff_counts[d] = c - 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Apply additions
        for d in added_diffs:
            c = diff_counts.get(d, 0)
            if c == 1:
                delta_collisions += 1
            diff_counts[d] = c + 1
            
        # Calculate new max
        new_row_max = max(row[:i] + [new_val] + row[i+1:])
        # Update global max
        # If old row was the max, and new row max is smaller, we need to check other rows.
        # If new row max is larger, global max is new row max.
        
        # Optimization: Keep a sorted list of row_maxs? Or just re-calculate max(row_maxs) if it changes?
        # n_rows is small (4). Just re-calculate.
        
        # Temporarily update row_maxs
        old_row_max = row_maxs[r_idx]
        row_maxs[r_idx] = new_row_max
        new_global_max = max(row_maxs)
        
        # Calculate cost
        new_collision_count = collision_count + delta_collisions
        new_cost = new_collision_count * 1000000 + new_global_max
        delta_cost = new_cost - current_cost
        
        # Acceptance
        accepted = False
        if delta_cost <= 0:
            accepted = True
        else:
            # SA acceptance
            if T > 0:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    accepted = True
        
        if accepted:
            # Commit changes
            row[i] = new_val
            collision_count = new_collision_count
            current_cost = new_cost
            # row_maxs and global_max already updated in calc
            # diff_counts already updated
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in rows]
                
        else:
            # Revert changes
            row[i] = old_val
            row_maxs[r_idx] = old_row_max
            # Revert diff_counts
            for d, c in orig_counts.items():
                if c == 0:
                    if d in diff_counts:
                        del diff_counts[d]
                else:
                    diff_counts[d] = c
            collision_count -= delta_collisions
            
        # Cool down
        T *= cooling
        
    # Return best valid solution found
    # Ensure best_rows is valid
    # If best_rows is not valid (should not happen if we tracked correctly), fallback to greedy
    # But we started with valid, and only kept best if cost decreased.
    # Cost includes collision count. So best_cost should have collision count 0 if we found a valid one.
    # If we didn't find a valid one better than greedy, best_rows is greedy (valid).
    
    # One edge case: if we accepted an invalid move temporarily, best_rows might be invalid?
    # No, we only update best_rows if current_cost < best_cost.
    # current_cost includes collision penalty.
    # So if best_rows was valid (cost = max_val), and we find an invalid one with lower max_val but high collision cost,
    # cost = 10^6 * C + M.
    # If C > 0, cost > 10^6.
    # Greedy max_val is likely < 10^6.
    # So invalid solutions won't replace valid best_rows.
    
    return best_rows
