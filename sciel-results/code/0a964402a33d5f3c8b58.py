import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Helper to check validity and get cost
    def get_cost(rows):
        used_diffs = set()
        max_entry = 0
        conflicts = 0
        for row in rows:
            if not row:
                continue
            max_entry = max(max_entry, row[-1])
            # Compute differences for this row
            row_diffs = []
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        conflicts += 1
                    else:
                        used_diffs.add(d)
                    row_diffs.append(d)
        # Cost is max entry + penalty for conflicts
        # We want to minimize max entry, but conflicts must be 0 for validity
        # Using a large penalty ensures we prioritize validity first, then minimize max entry
        return max_entry + 1000 * conflicts, max_entry, conflicts

    # Initial Greedy Solution
    def greedy_init():
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
                    used.update(new_diffs)
            rows.append(row)
        return rows

    current_rows = greedy_init()
    current_cost, current_max, current_conflicts = get_cost(current_rows)
    
    best_rows = [r[:] for r in current_rows]
    best_cost = current_cost
    best_max = current_max

    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 1.0
    
    # Random seed for reproducibility in local tests, but random in eval is fine
    random.seed(42)

    while time.time() < deadline and T > min_T:
        # Generate a neighbor
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a random element to change (index 1 to k)
        # Changing index 0 is not allowed (must be 0)
        elem_idx = random.randint(1, k)
        
        # Determine range for new value
        # Must be > row[elem_idx - 1] and < row[elem_idx + 1] (if exists)
        lower = row[elem_idx - 1] + 1
        upper = row[elem_idx + 1] - 1 if elem_idx < k else current_max + 50
        
        if lower > upper:
            T *= cooling
            continue
            
        # Bias towards smaller values to reduce max entry, but allow exploration
        # If we are stuck with conflicts, we might need to expand.
        # Simple strategy: pick uniformly from [lower, upper]
        # But if we want to shrink, we can bias towards lower.
        # Let's try a mix: 70% chance to pick from [lower, min(upper, row[elem_idx] + 10)]
        # 30% chance to pick from [lower, upper]
        if random.random() < 0.7:
            new_val = random.randint(lower, min(upper, row[elem_idx] + 10))
        else:
            new_val = random.randint(lower, upper)
            
        if new_val == row[elem_idx]:
            continue

        # Apply move
        old_val = row[elem_idx]
        row[elem_idx] = new_val
        
        # Check validity and cost
        new_cost, new_max, new_conflicts = get_cost(current_rows)
        
        # Acceptance criteria
        delta = new_cost - current_cost
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_cost = new_cost
            current_max = new_max
            current_conflicts = new_conflicts
            
            # Update best if valid (conflicts == 0) and better max
            if new_conflicts == 0 and new_max < best_max:
                best_max = new_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            row[elem_idx] = old_val
            
        T *= cooling

    # If we didn't find a valid solution better than greedy, return greedy
    # Actually, greedy is always valid.
    # If SA found a valid one with smaller max, return that.
    # If SA ended with conflicts, we should return the best valid one found.
    # best_rows is always valid because we only update it if conflicts == 0.
    
    return best_rows
