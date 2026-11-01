import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    
    def time_left():
        return deadline - time.time()

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_all_diffs(rows):
        """Returns a set of all differences in the configuration."""
        diffs = set()
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    diffs.add(r[j] - r[i])
        return diffs

    def is_valid(rows):
        """Check if all differences are distinct."""
        seen = set()
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_init():
        """
        Generate a valid initial configuration using a greedy approach.
        We build rows one by one.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                # Try candidates from last+1 upwards
                # We add a bit of randomness to avoid deterministic bias
                # but keep it small to ensure validity
                found = False
                # Look in a window
                for x in range(last + 1, last + 100):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        # Check internal row diffs (already done via 'a in row' which includes all previous)
                        # Wait, 'a in row' covers all previous elements.
                        # So new_diffs are distinct from each other?
                        # Yes, because x is fixed and a are distinct, so x-a are distinct.
                        row.append(x)
                        for d in new_diffs:
                            used_diffs.add(d)
                        found = True
                        break
                
                if not found:
                    # If window fails, extend search
                    # This should be rare for small M
                    x = last + 100
                    while True:
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                            new_diffs.append(d)
                        if valid:
                            row.append(x)
                            for d in new_diffs:
                                used_diffs.add(d)
                            break
                        x += 1
            
            rows.append(row)
            
        return rows

    def targeted_descent(rows, max_iterations=1000):
        """
        Refine the configuration by trying to reduce individual elements.
        We prioritize elements that are large.
        """
        rows = [list(r) for r in rows]
        best_scope = get_scope(rows)
        current_scope = best_scope
        
        # We will loop until no improvement is found or max iterations
        # Since the state space is discrete and small, this should converge fast.
        
        improved = True
        while improved and max_iterations > 0 and time_left() > 2:
            improved = False
            max_iterations -= 1
            
            # Find the element with the largest value
            # Actually, we can try to improve ANY element.
            # But reducing the largest is most likely to reduce scope.
            # Let's iterate through all elements in random order, or descending value order.
            
            # Get all elements with their (row, col)
            candidates = []
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len): # Skip 0
                    val = rows[r_idx][c_idx]
                    candidates.append((val, r_idx, c_idx))
            
            # Sort by value descending to target bottlenecks
            candidates.sort(reverse=True)
            
            for val, r_idx, c_idx in candidates:
                if time_left() < 2:
                    break
                
                # Try to reduce rows[r_idx][c_idx]
                # Get all diffs EXCEPT those involving this element
                # This is expensive to recompute every time.
                # Optimization: Maintain a global diff count or set?
                # Given small M (~134) and n_rows=4, total diffs ~ 4 * 28 = 112.
                # Recomputing is O(112). Fast enough.
                
                # Build set of "fixed" diffs (diffs not involving the target element)
                fixed_diffs = set()
                target_row = rows[r_idx]
                
                for r in rows:
                    for i in range(row_len):
                        for j in range(i + 1, row_len):
                            # Skip if this diff involves the target element
                            if r is target_row and (i == c_idx or j == c_idx):
                                continue
                            fixed_diffs.add(r[j] - r[i])
                
                # Now try to find a new value x' for target_row[c_idx]
                # Constraints:
                # 1. target_row[c_idx-1] < x' < target_row[c_idx+1] (if c_idx < row_len-1)
                #    If c_idx == row_len-1, x' > target_row[c_idx-1]
                # 2. All new diffs (x' - a) for a in target_row[:c_idx] must not be in fixed_diffs
                # 3. All new diffs (b - x') for b in target_row[c_idx+1:] must not be in fixed_diffs
                # 4. New diffs must be distinct from each other (they are, as argued before)
                # 5. New diffs must be distinct from each other within the row (handled by 2 & 3 if fixed_diffs is empty? No, fixed_diffs excludes target. But new diffs must not collide with EACH OTHER? Yes, they are distinct. Must not collide with FIXED diffs? Yes.)
                
                lower_bound = target_row[c_idx - 1] + 1
                upper_bound = target_row[c_idx] - 1
                if c_idx == row_len - 1:
                    # No upper bound from right neighbor, but we are trying to REDUCE, so we only look down.
                    # We want x' < current val.
                    pass
                else:
                    # If c_idx < row_len - 1, x' must be < target_row[c_idx+1]
                    # But we are looking for x' < target_row[c_idx].
                    # So upper_bound is effectively target_row[c_idx] - 1.
                    # We just need to ensure x' > lower_bound.
                    pass
                
                # We want to find the LARGEST valid x' that is < current val?
                # Or just ANY valid x' < current val?
                # To minimize scope, we want the smallest possible max.
                # If we reduce this element, does the scope drop?
                # If this element was the unique global max, yes.
                # If not, reducing it might not drop scope, but it might allow OTHER elements to be reduced later.
                # So we should accept ANY reduction.
                # To be most effective, let's try to reduce it as much as possible?
                # Or just find the first valid one downwards?
                # Finding the first valid one downwards is fast.
                
                found_new_val = False
                # Scan downwards from current val - 1 to lower_bound
                for new_val in range(target_row[c_idx] - 1, lower_bound - 1, -1):
                    valid = True
                    
                    # Check left neighbors
                    for i in range(c_idx):
                        d = new_val - target_row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check right neighbors
                    for j in range(c_idx + 1, row_len):
                        d = target_row[j] - new_val
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # If we reach here, new_val is valid
                    target_row[c_idx] = new_val
                    found_new_val = True
                    # Update scope if this was the max
                    new_scope = get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    # We accept this improvement and break the loop for this element
                    # We should probably restart the candidate sorting or just continue to next element
                    # Since we modified the row, the "fixed_diffs" for other elements is now stale.
                    # But we are in a loop. Let's just mark improved and break out to re-sort.
                    improved = True
                    break
            
            if improved:
                # We made a change, so we restart the scan
                continue
        
        return rows, best_scope

    def kick(rows, temperature=5.0):
        """
        Perturb the solution to escape local minima.
        Pick a random element and try to move it to a random valid position.
        Accept if better, or with probability exp(-delta/T) if worse.
        """
        rows = [list(r) for r in rows]
        current_scope = get_scope(rows)
        
        # Pick random row and col
        r_idx = random.randint(0, n_rows - 1)
        c_idx = random.randint(1, row_len - 1)
        
        # Determine valid range for new value
        lower = rows[r_idx][c_idx - 1] + 1
        upper = rows[r_idx][c_idx + 1] - 1 if c_idx < row_len - 1 else current_scope + 50
        
        if lower >= upper:
            return rows
        
        # Try a few random proposals
        for _ in range(5):
            new_val = random.randint(lower, upper)
            if new_val == rows[r_idx][c_idx]:
                continue
            
            # Check validity
            # Build fixed diffs (excluding this element)
            fixed_diffs = set()
            target_row = rows[r_idx]
            for r in rows:
                for i in range(row_len):
                    for j in range(i + 1, row_len):
                        if r is target_row and (i == c_idx or j == c_idx):
                            continue
                        fixed_diffs.add(r[j] - r[i])
            
            valid = True
            for i in range(c_idx):
                d = new_val - target_row[i]
                if d in fixed_diffs:
                    valid = False
                    break
            if not valid:
                continue
            
            for j in range(c_idx + 1, row_len):
                d = target_row[j] - new_val
                if d in fixed_diffs:
                    valid = False
                    break
            if not valid:
                continue
            
            # Valid move
            old_val = rows[r_idx][c_idx]
            rows[r_idx][c_idx] = new_val
            new_scope = get_scope(rows)
            
            if new_scope <= current_scope:
                return rows
            else:
                # Metropolis acceptance
                delta = new_scope - current_scope
                if random.random() < math.exp(-delta / temperature):
                    return rows
                else:
                    # Reject
                    rows[r_idx][c_idx] = old_val
        
        return rows

    # Main Logic
    best_rows = greedy_init()
    best_scope = get_scope(best_rows)
    
    # Initial Polish
    polished_rows, polished_scope = targeted_descent(best_rows)
    if polished_scope < best_scope:
        best_rows = polished_rows
        best_scope = polished_scope
        
    # ILS Loop
    temperature = 10.0
    while time_left() > 5:
        # Polish
        polished_rows, polished_scope = targeted_descent(best_rows, max_iterations=50)
        if polished_scope < best_scope:
            best_rows = polished_rows
            best_scope = polished_scope
            
        if time_left() < 2:
            break
            
        # Kick
        kicked_rows = kick(best_rows, temperature)
        
        # Polish the kicked solution
        polished_kicked_rows, polished_kicked_scope = targeted_descent(kicked_rows, max_iterations=50)
        
        if polished_kicked_scope < best_scope:
            best_rows = polished_kicked_rows
            best_scope = polished_kicked_scope
        else:
            # Even if not better, we might want to keep the kicked state if it's "equally good" but different?
            # No, we want to minimize scope. If it's not better, revert to best.
            # But maybe the kick led to a plateau. 
            # To allow exploration, we could accept equal scope with some probability?
            # For now, stick to strict improvement for the "best" record, but the "current" state for kicking should be the one we kicked from.
            # Actually, ILS usually keeps the current state if the kick+polish doesn't improve, to avoid drifting.
            # Or it reverts. Let's revert to best if not improved.
            pass
            
        # Cool down temperature
        temperature *= 0.95
        if temperature < 1.0:
            temperature = 1.0

    return best_rows
