import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Greedy initialization
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = len(row)  # start searching from a reasonable point
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
            rows.append(row)
        return rows
    
    # Compute all differences for a row
    def row_diffs(row):
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(row[j] - row[i])
        return diffs
    
    # Initialize
    init_rows = greedy_init()
    best_rows = [r[:] for r in init_rows]
    best_max = max(max(r) for r in best_rows)
    current_rows = [r[:] for r in init_rows]
    
    # Track difference counts efficiently
    # diff_count[d] = number of times difference d appears across all rows
    diff_count = {}
    for row in current_rows:
        for d in row_diffs(row):
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def count_violations():
        v = 0
        for c in diff_count.values():
            if c > 1:
                v += c - 1
        return v
    
    cur_viol = count_violations()
    cur_max = max(max(r) for r in current_rows)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.01
    # Cooling: reduce T gradually
    total_steps_est = 200000
    cooling = (T_min / T) ** (1.0 / total_steps_est)
    
    # Precompute all (row, pos) pairs for perturbation
    all_positions = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    step = 0
    while time.time() < deadline:
        # Pick random position
        r, i = random.choice(all_positions)
        old_val = current_rows[r][i]
        
        # Determine valid range
        lower = current_rows[r][i-1] + 1
        if i < k:
            upper = current_rows[r][i+1] - 1
        else:
            upper = cur_max + 50  # allow some growth
        
        if upper < lower:
            continue
        
        # Generate new value
        if random.random() < 0.15:
            # Large jump
            new_val = random.randint(lower, upper)
        else:
            # Small step with bias toward smaller values
            step_size = random.randint(-5, 5)
            new_val = old_val + step_size
            if new_val < lower:
                new_val = lower
            if new_val > upper:
                new_val = upper
        
        if new_val == old_val:
            continue
        
        # Apply move and update diff_count
        old_row = current_rows[r]
        
        # Remove old differences involving position i
        for j in range(k + 1):
            if j == i:
                continue
            if j < i:
                d = old_val - old_row[j]
            else:
                d = old_row[j] - old_val
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        # Set new value
        current_rows[r][i] = new_val
        new_row = current_rows[r]
        
        # Add new differences involving position i
        for j in range(k + 1):
            if j == i:
                continue
            if j < i:
                d = new_val - new_row[j]
            else:
                d = new_row[j] - new_val
            diff_count[d] = diff_count.get(d, 0) + 1
        
        # Compute new metrics
        new_viol = count_violations()
        new_max = max(max(row) for row in current_rows)
        
        # Cost function: violations are very expensive
        old_cost = cur_viol * 10000 + cur_max
        new_cost = new_viol * 10000 + new_max
        
        # Accept or reject
        if new_cost <= old_cost:
            # Accept
            cur_viol = new_viol
            cur_max = new_max
            if new_viol == 0 and new_max < best_max:
                best_max = new_max
                best_rows = [row[:] for row in current_rows]
        else:
            # Metropolis acceptance
            delta = new_cost - old_cost
            if random.random() < pow(2.718281828, -delta / T):
                cur_viol = new_viol
                cur_max = new_max
            else:
                # Reject: undo the move
                # Remove new differences
                for j in range(k + 1):
                    if j == i:
                        continue
                    if j < i:
                        d = new_val - new_row[j]
                    else:
                        d = new_row[j] - new_val
                    diff_count[d] -= 1
                    if diff_count[d] == 0:
                        del diff_count[d]
                
                # Restore old value
                current_rows[r][i] = old_val
                old_row = current_rows[r]
                
                # Add old differences back
                for j in range(k + 1):
                    if j == i:
                        continue
                    if j < i:
                        d = old_val - old_row[j]
                    else:
                        d = old_row[j] - old_val
                    diff_count[d] = diff_count.get(d, 0) + 1
        
        # Cool down
        step += 1
        if step % 1000 == 0:
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Return best found
    return best_rows
# EVOLVE-BLOCK-END
