import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_diff_counts(rows):
        """
        Returns (diff_counts_dict, total_conflicts)
        diff_counts: {diff_value: count}
        total_conflicts: sum of C(count, 2) for all diffs
        """
        counts = defaultdict(int)
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] += 1
        
        total_conf = 0
        for d, c in counts.items():
            if c > 1:
                total_conf += c * (c - 1) // 2
        return counts, total_conf

    def greedy_init(n_rows, k, max_val=200, seed=0):
        """
        Generate a valid initial solution using greedy.
        """
        random.seed(seed)
        used = set()
        rows = []
        for i in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                # Try values from last+1 to max_val
                found = False
                for x in range(last + 1, max_val + 1):
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    # Phase 1: Initialization
    best_rows = None
    best_scope = float('inf')
    
    # Try a few greedy inits to get a good starting point
    for seed in range(10):
        if time.time() > deadline - 120:
            break
        rows = greedy_init(n_rows, k, 200, seed)
        if rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Emergency fallback: just return a valid greedy one, even if scope is large
        # This should not happen with max_val=200 for n=4, k=7
        return greedy_init(n_rows, k, 500, 0)

    # Phase 2: Variable-Level Simulated Annealing (VLSA)
    # State: rows, diff_counts, total_conflicts, scope
    current_rows = [r[:] for r in best_rows]
    current_counts, current_conflicts = compute_diff_counts(current_rows)
    current_scope = best_scope
    
    # Cost function: Scope is primary, Conflicts are secondary (heavy penalty)
    # We want to minimize Scope, but we must end with 0 conflicts.
    # Cost = Scope * 1000 + Conflicts
    # This encourages reducing scope, but heavily penalizes conflicts.
    # However, if we are stuck with conflicts, we might need to increase scope to resolve them.
    # A better cost: If conflicts > 0, Cost = Conflicts * 10000 + Scope.
    # If conflicts == 0, Cost = Scope.
    # This ensures we prioritize finding *any* valid solution, then minimize scope.
    
    def get_cost(scope, conflicts):
        if conflicts > 0:
            return conflicts * 10000 + scope
        return scope

    current_cost = get_cost(current_scope, current_conflicts)
    best_valid_rows = [r[:] for r in best_rows]
    best_valid_scope = best_scope
    best_valid_cost = get_cost(best_valid_scope, 0) # Init is valid

    temp = 50.0
    cooling = 0.999
    steps = 0
    max_steps = 500000
    
    # Pre-calculate max_val for sampling
    # We can sample new values in a window around current value, or random in [prev+1, next-1]
    
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        
        # Pick a random row and index (skip index 0 which is always 0)
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, row_len - 1)
        
        row = current_rows[r_idx]
        old_val = row[i_idx]
        
        # Determine valid range for new value
        min_val = row[i_idx - 1] + 1
        max_val = row[i_idx + 1] - 1 if i_idx < row_len - 1 else 200 # Upper bound heuristic
        
        if min_val > max_val:
            temp *= cooling
            continue
            
        # Sample new value
        # Strategy: 50% local move (small change), 50% global random in range
        if random.random() < 0.5:
            # Local move: +/- 1, 2, 5
            delta = random.choice([-5, -2, -1, 1, 2, 5])
            new_val = old_val + delta
            if new_val < min_val: new_val = min_val
            if new_val > max_val: new_val = max_val
        else:
            new_val = random.randint(min_val, max_val)
            
        if new_val == old_val:
            continue
            
        # Tentative update
        # 1. Update diff counts
        # Affected diffs are those involving row[r_idx][i_idx]
        # i.e., abs(new_val - row[r_idx][j]) for all j != i_idx
        
        # We need to track the change in total_conflicts
        delta_conflicts = 0
        
        # Store old counts for affected diffs to restore if rejected
        # Or just apply and reverse.
        # Let's apply and keep track of changes.
        
        changes = [] # List of (diff, delta_count)
        
        for j in range(row_len):
            if j == i_idx:
                continue
            other_val = row[j]
            old_d = abs(old_val - other_val)
            new_d = abs(new_val - other_val)
            
            if old_d != new_d:
                # Decrement old_d
                old_count = current_counts[old_d]
                # Update conflict contribution for old_d
                old_pairs = old_count * (old_count - 1) // 2
                new_count_old = old_count - 1
                new_pairs_old = new_count_old * (new_count_old - 1) // 2
                delta_conflicts += new_pairs_old - old_pairs
                
                current_counts[old_d] = new_count_old
                changes.append((old_d, -1))
                
                # Increment new_d
                old_count_new = current_counts[new_d]
                old_pairs_new = old_count_new * (old_count_new - 1) // 2
                new_count_new = old_count_new + 1
                new_pairs_new = new_count_new * (new_count_new - 1) // 2
                delta_conflicts += new_pairs_new - old_pairs_new
                
                current_counts[new_d] = new_count_new
                changes.append((new_d, 1))
        
        new_conflicts = current_conflicts + delta_conflicts
        
        # Update Scope
        # Only the max of the specific row might change, and global max might change
        # Simple: recalc scope of the modified row, then global max
        row_max = max(row[:i_idx] + [new_val] + row[i_idx+1:])
        
        # To avoid full scan for global scope every step, we can keep a global max heap or just scan.
        # Scanning 4 rows of 8 ints is very fast.
        
        new_scope = 0
        for r_idx_check in range(n_rows):
            r = current_rows[r_idx_check]
            if r_idx_check == r_idx:
                # Use new_val
                # Find max of row with new_val at i_idx
                m = 0
                for j in range(row_len):
                    v = new_val if j == i_idx else r[j]
                    if v > m: m = v
                if m > new_scope: new_scope = m
            else:
                if max(r) > new_scope:
                    new_scope = max(r)
        
        new_cost = get_cost(new_scope, new_conflicts)
        
        # Acceptance
        delta_cost = new_cost - current_cost
        
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / temp):
            # Accept
            current_rows[r_idx][i_idx] = new_val
            current_conflicts = new_conflicts
            current_scope = new_scope
            current_cost = new_cost
            
            # Update best valid
            if new_conflicts == 0 and new_scope < best_valid_scope:
                best_valid_scope = new_scope
                best_valid_rows = [r[:] for r in current_rows]
                # Optional: reset temp or boost if we found a new best?
                # Keep cooling.
                
        else:
            # Reject: Revert diff counts
            for d, delta in changes:
                current_counts[d] -= delta
            # Note: current_conflicts, current_scope, current_cost remain unchanged
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1 # Floor temperature to allow some exploration
            
    # Return best valid found
    if not validate(best_valid_rows):
        # Should not happen if init was valid and we only accept valid states as best
        # But just in case
        return greedy_init(n_rows, k, 500, 0)
        
    return best_valid_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
