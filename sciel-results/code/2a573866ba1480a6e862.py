import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: m*(m+1)/2 >= total_diffs_needed
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
        
    # Greedy baseline
    def greedy_construct(n_rows, k):
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # SA State
    def init_state(rows):
        # rows is list of lists
        diff_counts = {}
        collisions = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in diff_counts:
                        diff_counts[d] += 1
                        if diff_counts[d] == 2:
                            collisions += 1
                    else:
                        diff_counts[d] = 1
        max_val = max(max(r) for r in rows)
        return diff_counts, collisions, max_val

    def energy(collisions, max_val):
        return 100 * collisions + max_val

    current_rows = [list(r) for r in greedy_rows]
    current_diff_counts, current_collisions, current_max = init_state(current_rows)
    current_energy = energy(current_collisions, current_max)
    
    best_rows = [list(r) for r in current_rows]
    best_energy = current_energy
    best_max = current_max
    
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < time_limit - 2.0:
        # Generate move
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, k)  # 1 to k (inclusive), since index 0 is 0
        
        row = current_rows[r_idx]
        old_v = row[i_idx]
        
        # Determine valid range for new_v
        low = row[i_idx - 1] + 1
        high = row[i_idx + 1] - 1 if i_idx < k else current_max + 50 # Allow expansion? No, we want to minimize.
        # Actually, if we allow expansion, we might escape local minima.
        # But we want to minimize max.
        # Let's allow high to be max(current_max, old_v + 10) to allow some flexibility, but bias towards low.
        
        # If low > high, move is invalid
        if low > high:
            continue
            
        # Bias towards smaller values to minimize max
        # Sample uniformly in [low, high]
        new_v = random.randint(low, high)
        
        if new_v == old_v:
            continue
            
        # Calculate delta energy
        # Remove old diffs
        delta_collisions = 0
        diffs_to_remove = []
        for j in range(i_idx):
            d = old_v - row[j]
            diffs_to_remove.append(d)
        for j in range(i_idx + 1, len(row)):
            d = row[j] - old_v
            diffs_to_remove.append(d)
            
        # Add new diffs
        diffs_to_add = []
        for j in range(i_idx):
            d = new_v - row[j]
            diffs_to_add.append(d)
        for j in range(i_idx + 1, len(row)):
            d = row[j] - new_v
            diffs_to_add.append(d)
            
        # Apply changes to diff_counts temporarily to calculate new collisions
        # We need to be careful with the collision count update
        
        old_counts_snapshot = {}
        # Since we only change a few diffs, we can just update the counts and track collision changes
        
        # Helper to update collision count when count of a diff changes
        def update_collision(delta, d):
            nonlocal current_collisions
            # This is tricky to do inline without modifying state.
            # Let's just calculate new collisions from scratch? No, too slow?
            # 144 diffs total. Recalculating is O(144).
            # Move is O(15).
            # Recalculating is simpler and robust.
            pass
            
        # Let's try the recalculation approach for simplicity and correctness first.
        # If it's too slow, we optimize.
        # 144 diffs * 100k steps = 14.4M ops. Python might be slow.
        # Let's optimize the collision update.
        
        # We will manually track the change in collisions.
        # For each diff in diffs_to_remove:
        #   count decreases by 1
        #   if count was 2, collisions -= 1
        #   if count was 1, collisions unchanged
        # For each diff in diffs_to_add:
        #   count increases by 1
        #   if count was 1, collisions += 1
        #   if count was 0, collisions unchanged
        
        # But we have to handle if a diff is in both remove and add lists (cancellation).
        # Better: 
        # 1. Identify unique diffs affected.
        # 2. Calculate net change in count for each.
        # 3. Update collision count based on old count and new count.
        
        affected_diffs = {}
        for d in diffs_to_remove:
            affected_diffs[d] = affected_diffs.get(d, 0) - 1
        for d in diffs_to_add:
            affected_diffs[d] = affected_diffs.get(d, 0) + 1
            
        delta_coll = 0
        for d, delta_count in affected_diffs.items():
            if delta_count == 0:
                continue
            old_count = current_diff_counts.get(d, 0)
            new_count = old_count + delta_count
            if new_count < 0:
                # Should not happen if logic is correct
                continue
                
            # Update collision count
            if old_count >= 2 and new_count < 2:
                delta_coll -= 1
            elif old_count < 2 and new_count >= 2:
                delta_coll += 1
                
        new_collisions = current_collisions + delta_coll
        
        # Update max
        new_max = current_max
        if new_v > new_max:
            new_max = new_v
        elif old_v == new_max and new_v < old_v:
            # Need to recalculate max
            # Check if any other value is >= new_v
            # Simplest: max of all rows
            # To save time, check if any value in the modified row or other rows is > new_v
            # Actually, just recalculate if old_v was max.
            # Cost: 36 ops.
            new_max = max(max(r) for r in current_rows)
            # Note: current_rows still has old_v. So we need to account for the change.
            # Let's just set row[i_idx] = new_v temporarily? No, we don't want to modify state if we reject.
            # We can calculate max of (current_rows with row[i_idx] replaced by new_v).
            # Or just:
            max_others = max(max(r) for r in current_rows if r is not row)
            max_this_row = max(row[:i_idx] + [new_v] + row[i_idx+1:])
            new_max = max(max_others, max_this_row)
            
        new_energy = energy(new_collisions, new_max)
        
        # Acceptance
        delta_e = new_energy - current_energy
        accept = False
        if delta_e <= 0:
            accept = True
        else:
            if T > 0:
                prob = 2.718281828459045 ** (-delta_e / T)
                if random.random() < prob:
                    accept = True
            
        if accept:
            # Apply move
            row[i_idx] = new_v
            
            # Update diff_counts
            for d, delta_count in affected_diffs.items():
                if delta_count == 0:
                    continue
                current_diff_counts[d] = current_diff_counts.get(d, 0) + delta_count
                if current_diff_counts[d] == 0:
                    del current_diff_counts[d]
                    
            current_collisions = new_collisions
            current_max = new_max
            current_energy = new_energy
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_max = current_max
                best_rows = [list(r) for r in current_rows]
                
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: Restart with higher T if stuck?
            # For now, just keep going.
            
    # Return best valid solution
    # Ensure best_rows is valid
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
        
    if not validate(best_rows, n_rows, k):
        # Fallback to greedy
        return greedy_rows
        
    return best_rows
