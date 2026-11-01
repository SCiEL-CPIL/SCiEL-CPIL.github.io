import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = k + 1
    total_diffs_per_row = n * (n - 1) // 2
    
    # 1. Generate an initial valid solution using a greedy approach
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_score = max(max(r) for r in best_rows)
    
    # 2. Define evaluation function
    def evaluate(rows):
        used = set()
        max_entry = 0
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
                    if row[j] > max_entry:
                        max_entry = row[j]
        # We want to minimize max_entry, but conflicts are fatal.
        # Score: max_entry + conflicts * penalty
        penalty = 1000
        return max_entry + conflicts * penalty, conflicts, max_entry

    current_rows = [row[:] for row in best_rows]
    current_eval, current_conflicts, current_max = evaluate(current_rows)
    best_eval, best_conflicts, best_max = evaluate(best_rows)
    
    # 3. Simulated Annealing
    T = 50.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Move: Shift suffix of a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Choose split point
        s_idx = random.randint(1, n - 1)
        
        # Choose delta
        delta = random.randint(-10, 10)
        
        # Check validity of shift
        # If shifting right (delta > 0), ensure it doesn't exceed current max? 
        # Actually, we want to minimize max, so shifting right might increase it.
        # If shifting left (delta < 0), ensure row[s-1] < row[s] + delta
        
        if s_idx > 0:
            if row[s_idx] + delta <= row[s_idx - 1]:
                # Try to adjust delta to be minimal valid increase
                min_delta = row[s_idx - 1] - row[s_idx] + 1
                if min_delta > 0:
                    delta = min_delta
                else:
                    continue # Cannot shift left
            
            # If shifting right, no lower bound issue, but it increases max potentially.
            # We rely on SA to reject if it increases max too much or creates conflicts.
        
        # Apply move
        for j in range(s_idx, n):
            current_rows[r_idx][j] += delta
        
        new_eval, new_conflicts, new_max = evaluate(current_rows)
        
        # Acceptance criterion
        if new_eval < current_eval:
            current_eval = new_eval
            current_conflicts = new_conflicts
            current_max = new_max
            if new_eval < best_eval:
                best_eval = new_eval
                best_conflicts = new_conflicts
                best_max = new_max
                best_rows = [row[:] for row in current_rows]
        else:
            delta_eval = new_eval - current_eval
            if random.random() < pow(2.71828, -delta_eval / T):
                current_eval = new_eval
                current_conflicts = new_conflicts
                current_max = new_max
            else:
                # Revert
                for j in range(s_idx, n):
                    current_rows[r_idx][j] -= delta
        
        T *= cooling_rate

    # Return best valid solution found
    if best_conflicts == 0:
        return best_rows
    else:
        # Fallback to greedy if somehow invalid (should not happen if greedy is valid)
        return greedy_construct()
