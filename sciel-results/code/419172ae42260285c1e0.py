"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    def greedy_row(used):
        """Build one row greedily, returning the row and updated used set."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row, used

    def try_construction():
        """Try constructing rows sequentially."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row, used = greedy_row(used)
            rows.append(row)
        max_val = max(max(r) for r in rows)
        return rows, max_val

    # Strategy 1: Standard greedy to get a valid starting point
    rows, best_max = try_construction()
    
    # If we already have a perfect or near-perfect score, return immediately
    # (Though we don't know the target, 112 is the known best for 7,5)
    # For general n, k, we just proceed to SA.
    
    # Strategy 2: Simulated Annealing
    # Objective: Minimize max(max(r) for r in rows)
    
    current_rows = [r[:] for r in rows]
    current_max = best_max
    best_rows = [r[:] for r in rows]
    best_score = best_max
    
    start_time = time.time()
    initial_temp = 50.0
    cooling_rate = 0.999
    
    # Precompute differences for current rows to speed up checking?
    # Actually, rebuilding a row is O(k * max_val). Checking validity is O(k^2).
    # Rebuilding involves scanning x.
    
    def get_used_diffs(rows_except_idx):
        """Calculate the set of used differences from all rows except the one at idx."""
        used = set()
        for i, r in enumerate(rows_except_idx):
            if i == idx:
                continue
            for j in range(len(r)):
                for l in range(j+1, len(r)):
                    used.add(r[l] - r[j])
        return used

    # Optimization: Instead of recalculating all diffs, maintain a global count or set.
    # But rebuilding one row means removing its diffs and adding new ones.
    # Let's just recalculate the 'forbidden' set for the target row.
    
    def calculate_forbidden(rows, target_idx):
        forbidden = set()
        for i, r in enumerate(rows):
            if i == target_idx:
                continue
            for j in range(len(r)):
                for l in range(j+1, len(r)):
                    forbidden.add(r[l] - r[j])
        return forbidden

    def try_rebuild_row(row_idx, forbidden):
        """
        Attempt to rebuild the row at row_idx to minimize its max value.
        Returns the new row if it is valid and better (or potentially better for SA), else None.
        """
        # We want to find a row with small max value that doesn't use forbidden diffs.
        # Greedy approach: Start with 0, then pick smallest valid x.
        # This minimizes the max value for this specific row given the forbidden set.
        
        new_row = [0]
        x = 0
        local_used = set()
        
        while len(new_row) < k + 1:
            x += 1
            new_diffs = {x - a for a in new_row}
            
            # Check if any of these diffs are forbidden or already used in this row
            # Note: new_diffs are always unique within the row because the row is strictly increasing
            # and we check against 'local_used' (which is the set of diffs in new_row so far)
            # Wait, new_diffs = {x - a for a in new_row}. Since new_row is sorted, these are distinct.
            # But we must ensure they don't conflict with 'forbidden' or each other?
            # They are distinct by construction.
            # We just need to ensure they don't conflict with 'forbidden'.
            
            if not (new_diffs & forbidden) and not (new_diffs & local_used):
                new_row.append(x)
                local_used |= new_diffs
            else:
                # If we get stuck, we might need backtracking, but greedy is fast.
                # If we exceed a reasonable bound, we might fail to complete.
                # However, since the original solution was valid, a solution likely exists.
                # If greedy fails to complete, we return None.
                if x > current_max + 20: 
                    return None
        return new_row

    temp = initial_temp
    step = 0
    
    while time.time() < deadline - 5:
        step += 1
        
        # Select a random row to modify
        idx = random.randint(0, n_rows - 1)
        
        # Calculate forbidden differences from other rows
        forbidden = calculate_forbidden(current_rows, idx)
        
        # Try to rebuild the row
        new_row = try_rebuild_row(idx, forbidden)
        
        if new_row is None:
            # Greedy failed to build a valid row with this forbidden set.
            # This might happen if the forbidden set is too restrictive for a small max.
            # We could try a slightly larger max, but for now, skip this move.
            temp *= cooling_rate
            continue
            
        new_max = max(new_row)
        
        # Calculate the new global max
        # We need to know if new_max is larger than the max of the other rows
        other_max = 0
        for i in range(n_rows):
            if i != idx:
                if max(current_rows[i]) > other_max:
                    other_max = max(current_rows[i])
        
        new_global_max = max(new_max, other_max)
        old_global_max = current_max
        
        delta = new_global_max - old_global_max
        
        if delta <= 0:
            # Accept
            current_rows[idx] = new_row
            current_max = new_global_max
            
            if current_max < best_score:
                best_score = current_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Probabilistic acceptance
            if temp > 0:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    current_rows[idx] = new_row
                    current_max = new_global_max
        
        temp *= cooling_rate
        
        # Periodically restart from best found to avoid getting stuck in bad local minima
        if step % 1000 == 0:
             if current_max > best_score:
                 # Revert to best
                 current_rows = [r[:] for r in best_rows]
                 current_max = best_score
             # Small chance to randomize temperature to escape
             if random.random() < 0.1:
                 temp = initial_temp * 0.5

    return best_rows
# EVOLVE-BLOCK-END
