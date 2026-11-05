"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0

    # --- Helper: compute all positive differences for a list of rows ---
    def all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(row) for row in rows)

    def compute_cost(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        violations = sum(c - 1 for c in counts.values() if c > 1)
        m = max_entry(rows)
        return violations * 100000 + m

    def get_row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # --- Phase 1: Greedy initialization ---
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    current_rows = [r[:] for r in best_rows]
    current_cost = compute_cost(current_rows)
    
    # Initialize global_diffs
    global_diffs = set(all_diffs(current_rows))

    # --- Rebuild Move: Inspired by Inspiration 1 ---
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit):
        """
        Tries to build a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        Returns the row or None.
        """
        row = [0]
        local_used = set() # Diffs within this new row
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append(x)
                    if len(valid_candidates) >= 5:
                        break
            
            if not valid_candidates:
                return None
            
            x = valid_candidates[rng.randint(0, len(valid_candidates)-1)]
            row.append(x)
            local_used.update({x - a for a in row[:-1]})
            
        return row

    # --- Local Perturb Move: From Current Program ---
    def local_perturb(rows, rng):
        r = rng.randint(0, n_rows - 1)
        i = rng.randint(1, k) # index 1..k
        old_val = rows[r][i]
        
        lower = rows[r][i - 1] + 1
        upper = rows[r][i + 1] - 1 if i < k else rows[r][k] + 20
        
        if lower > upper:
            return None, None, None
        
        if rng.random() < 0.5:
             new_val = rng.randint(lower, min(upper, lower + 10))
        else:
             new_val = rng.randint(lower, upper)
             
        if new_val == old_val:
            return None, None, None
            
        return r, i, new_val

    # --- Main SA Loop ---
    T = 100.0
    T_min = 1.0
    cooling = 0.9999
    p_rebuild = 0.7

    while time.time() < deadline:
        if T < T_min:
            T = T_min # Floor the temperature to prevent underflow/zero probability

        move_type = "rebuild" if rng.random() < p_rebuild else "perturb"
        
        new_rows = [r[:] for r in current_rows]
        accepted = False
        revert_needed = False
        r_idx = -1
        old_row_copy = None
        old_diffs_copy = None
        new_diffs_copy = None

        if move_type == "rebuild":
            r = rng.randint(0, n_rows - 1)
            old_row = current_rows[r]
            old_diffs = get_row_diffs(old_row)
            old_row_copy = old_row[:]
            old_diffs_copy = set(old_diffs)
            
            # Remove old diffs from global set
            global_diffs.difference_update(old_diffs)
            
            target_max = max_entry(current_rows) - 1
            if target_max < k:
                target_max = k
                
            new_row = rebuild_row(k, global_diffs, rng, target_max)
            
            if new_row is None:
                new_row = rebuild_row(k, global_diffs, rng, max_entry(current_rows))
                
            if new_row is None:
                # Rebuild failed, restore old diffs
                global_diffs.update(old_diffs)
                T *= cooling
                continue
            else:
                new_rows[r] = new_row
                new_diffs = get_row_diffs(new_row)
                global_diffs.update(new_diffs)
                new_diffs_copy = set(new_diffs)
                r_idx = r
                revert_needed = True # If rejected, we must revert global_diffs

        else: # perturb
            r, i, new_val = local_perturb(current_rows, rng)
            if r is None:
                T *= cooling
                continue
                
            if i > 0 and new_val <= current_rows[r][i-1]:
                T *= cooling
                continue
            if i < k and new_val >= current_rows[r][i+1]:
                T *= cooling
                continue
                
            new_rows[r][i] = new_val
            r_idx = r
            # No global_diffs update yet, will do on accept

        # Calculate new cost
        new_cost = compute_cost(new_rows)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            accepted = True
            current_rows = new_rows
            current_cost = new_cost
            
            # Update global_diffs if perturb
            if move_type == "perturb":
                global_diffs = set(all_diffs(new_rows))
                
            # Check if valid and better
            if current_cost < 100000: # No violations
                curr_max = max_entry(current_rows)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = [r[:] for r in current_rows]
        else:
            # Reject
            if move_type == "rebuild" and revert_needed:
                # Remove new diffs, restore old diffs
                global_diffs.difference_update(new_diffs_copy)
                global_diffs.update(old_diffs_copy)

        T *= cooling

    return best_rows
# EVOLVE-BLOCK-END
