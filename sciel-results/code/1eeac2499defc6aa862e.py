import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s. Keep margin.
    deadline = start_time + 110.0
    
    # Helper: Get differences for a row
    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: Greedy construction of a single row given forbidden diffs
    def greedy_row(forbidden_diffs, k, randomize=False, bias=0.1):
        row = [0]
        for _ in range(k):
            last = row[-1]
            valid_xs = []
            x = last + 1
            # Search for valid x
            # Limit search to avoid infinite loops in pathological cases, though rare for small k
            while x < 300: 
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden_diffs):
                    valid_xs.append(x)
                    if len(valid_xs) >= 10:
                        break
                x += 1
            
            if not valid_xs:
                return None
            
            if randomize:
                # Pick the smallest with high probability, otherwise random from small set
                if random.random() > bias:
                    row.append(valid_xs[0])
                else:
                    row.append(random.choice(valid_xs))
            else:
                row.append(valid_xs[0])
        return row

    # Helper: Get used diffs from a list of rows
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            used |= get_row_diffs(row)
        return used

    # 1. Initial Greedy Solution
    rows = []
    used_diffs = set()
    for _ in range(n_rows):
        row = greedy_row(used_diffs, k, randomize=False)
        if row is None:
            # Fallback, shouldn't happen for small k
            return rows
        rows.append(row)
        used_diffs |= get_row_diffs(row)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # 2. Simulated Annealing
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while time.time() < deadline:
        # Pick a random row to modify
        i = random.randint(0, n_rows - 1)
        old_row = rows[i]
        
        # Remove old row's diffs
        old_diffs = get_row_diffs(old_row)
        used_diffs -= old_diffs
        
        # Generate new row
        # We want to explore. Use randomization.
        # Bias can be adjusted.
        new_row = greedy_row(used_diffs, k, randomize=True, bias=0.2)
        
        if new_row is None:
            # Failed to generate, restore
            used_diffs |= old_diffs
            continue
        
        # Calculate Global Max Change
        # Current global max
        # Optimization: We can keep track of maxes, but n is small.
        old_global_max = max(max(rows[j]) for j in range(n_rows) if j != i)
        old_row_max = max(old_row)
        current_global_max = max(old_global_max, old_row_max)
        
        new_row_max = max(new_row)
        new_global_max = max(old_global_max, new_row_max)
        
        delta = new_global_max - current_global_max
        
        # Acceptance Criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            rows[i] = new_row
            used_diffs |= get_row_diffs(new_row)
            
            if new_global_max < best_max:
                best_max = new_global_max
                best_rows = [r[:] for r in rows]
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_rows
