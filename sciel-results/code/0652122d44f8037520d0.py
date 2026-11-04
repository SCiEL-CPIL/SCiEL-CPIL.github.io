import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # 5s margin from 120s limit
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def init_state():
        """Greedy initialization to get a valid starting point."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = init_state()
    
    # Build initial diff counts
    diff_counts = defaultdict(int)
    for r in range(n_rows):
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = rows[r][j] - rows[r][i]
                diff_counts[d] += 1
                
    def get_initial_metrics():
        unique_count = sum(1 for c in diff_counts.values() if c > 0)
        dups = total_diffs - unique_count
        scope = max(max(r) for r in rows)
        cost = dups * 1000 + scope
        return dups, scope, cost

    cur_dups, cur_scope, cur_cost = get_initial_metrics()
    best_rows = [r[:] for r in rows]
    best_cost = cur_cost
    
    # Simulated Annealing parameters
    T = 50.0
    cooling = 0.999
    min_T = 0.5
    
    # Helper to update diff counts and dups for a row change
    def apply_change(r, c, old_val, new_val):
        """
        Applies the change to rows and diff_counts.
        Returns the change in dups (delta_dups).
        """
        nonlocal cur_dups
        delta_dups = 0
        
        # Remove old differences
        for j in range(k + 1):
            if j == c:
                continue
            d = abs(rows[r][j] - old_val)
            old_count = diff_counts[d]
            diff_counts[d] = old_count - 1
            
            # Update dups
            if old_count == 1:
                # Was unique, now 0. Unique count -1 => Dups +1
                delta_dups += 1
            elif old_count > 1:
                # Was duplicate, now less duplicate. Dups -1
                delta_dups -= 1
            # else old_count == 0, shouldn't happen

        # Add new differences
        for j in range(k + 1):
            if j == c:
                continue
            d = abs(rows[r][j] - new_val)
            new_count = diff_counts[d]
            diff_counts[d] = new_count + 1
            
            # Update dups
            if new_count == 0:
                # Was 0, now 1. Unique count +1 => Dups -1
                delta_dups -= 1
            elif new_count >= 1:
                # Was present, now more duplicate. Dups +1
                delta_dups += 1
                
        cur_dups += delta_dups

    def revert_change(r, c, old_val, new_val):
        """
        Reverts the change.
        """
        nonlocal cur_dups
        delta_dups = 0
        
        # Remove new differences (which are currently in diff_counts)
        for j in range(k + 1):
            if j == c:
                continue
            d = abs(rows[r][j] - new_val)
            curr_count = diff_counts[d]
            diff_counts[d] = curr_count - 1
            
            if curr_count == 1:
                delta_dups += 1
            elif curr_count > 1:
                delta_dups -= 1

        # Add old differences
        for j in range(k + 1):
            if j == c:
                continue
            d = abs(rows[r][j] - old_val)
            curr_count = diff_counts[d]
            diff_counts[d] = curr_count + 1
            
            if curr_count == 0:
                delta_dups -= 1
            elif curr_count >= 1:
                delta_dups += 1

        cur_dups += delta_dups

    while time.time() - start_time < deadline:
        if T < min_T:
            T = 50.0 # Restart temperature if it gets too low to keep exploring
            
        r = random.randint(0, n_rows - 1)
        c = random.randint(1, k)
        
        old_val = rows[r][c]
        # Perturb with a small delta
        delta = random.randint(-10, 10)
        if delta == 0:
            continue
        new_val = old_val + delta
        
        # Check strictly increasing constraint
        if new_val <= 0:
            continue
        if c > 0 and new_val <= rows[r][c - 1]:
            continue
        if c < k and new_val >= rows[r][c + 1]:
            continue
            
        # Store old state
        old_dups = cur_dups
        old_scope = cur_scope
        old_cost = cur_cost
        
        # Apply change
        rows[r][c] = new_val
        apply_change(r, c, old_val, new_val)
        
        # Calculate new metrics
        new_scope = max(max(row) for row in rows)
        new_cost = cur_dups * 1000 + new_scope
        
        # Acceptance criterion
        delta_cost = new_cost - old_cost
        if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
            # Accept
            cur_scope = new_scope
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_rows = [row[:] for row in rows]
        else:
            # Reject
            rows[r][c] = old_val
            revert_change(r, c, old_val, new_val)
            cur_dups = old_dups
            cur_scope = old_scope
            cur_cost = old_cost
            
        T *= cooling

    return best_rows
