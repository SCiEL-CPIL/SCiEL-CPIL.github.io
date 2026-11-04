"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
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

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

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

    # --- Phase 2: Simulated annealing with incremental difference tracking ---
    current_rows = [r[:] for r in best_rows]
    
    # Precompute the set of "indices" we can perturb: (row_idx, pos_idx) where pos_idx in 1..k
    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    # Incremental difference tracking
    diff_counts = Counter()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff_counts[row[j] - row[i]] += 1
    
    def get_violations(diff_counts):
        return sum(c - 1 for c in diff_counts.values() if c > 1)
    
    current_violations = get_violations(diff_counts)
    current_max = best_max
    current_cost = current_violations * 10000 + current_max

    T = 50.0
    T_min = 0.01
    cooling = 0.99995

    while time.time() < deadline:
        if T < T_min:
            break

        # Pick a random perturbation
        r, i = random.choice(indices)
        old_val = current_rows[r][i]

        # Determine valid range for new value
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 50

        # Bias towards smaller values to reduce max
        if random.random() < 0.3:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.5)
        else:
            new_val = random.randint(lower, upper)

        if new_val == old_val:
            continue

        # Check strict increasing is still maintained
        if new_val <= lower or (i < k and new_val >= upper):
            continue

        # Compute differences that will be removed and added
        row = current_rows[r]
        removed_diffs = []
        added_diffs = []
        
        # Differences involving position i
        for j in range(len(row)):
            if j == i:
                continue
            if j < i:
                # row[i] - row[j]
                old_d = old_val - row[j]
                new_d = new_val - row[j]
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
            else:
                # row[j] - row[i]
                old_d = row[j] - old_val
                new_d = row[j] - new_val
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
        
        # Update diff_counts incrementally
        for d in removed_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        for d in added_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        new_violations = get_violations(diff_counts)
        new_max = max(max(row) for row in current_rows)
        new_cost = new_violations * 10000 + new_max

        # Simulated annealing acceptance
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept
            current_rows[r][i] = new_val
            current_violations = new_violations
            current_max = new_max
            current_cost = new_cost
            
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            # Reject: revert diff_counts
            for d in added_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in removed_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            current_rows[r][i] = old_val

        T *= cooling

    return best_rows
# EVOLVE-BLOCK-END
