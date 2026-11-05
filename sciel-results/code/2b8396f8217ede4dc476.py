import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs
        
    def greedy_init():
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(x)
                    used_diffs |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    def compute_all_diffs(rows):
        return [get_row_diffs(r) for r in rows]

    best_solution = greedy_init()
    best_diffs = compute_all_diffs(best_solution)
    best_max = max(max(r) for r in best_solution)
    
    current_solution = [list(r) for r in best_solution]
    current_diffs = [set(d) for d in best_diffs]
    current_max = best_max
    
    # Simulated Annealing
    T = 50.0  # Initial temperature
    cooling_rate = 0.995
    
    def get_new_val(row, pos):
        # Get valid range for row[pos]
        lower = row[pos-1] + 1
        upper = row[pos+1] - 1 if pos < k else current_max + 10 # Allow expansion? 
        # Actually, we want to minimize max, so we shouldn't expand too much.
        # But to escape minima, we might need to.
        # Let's cap it to current_max + 20 to allow some breathing room.
        upper = min(upper, current_max + 20)
        
        if lower > upper:
            return None
            
        # Bias towards smaller values?
        # Uniform random in [lower, upper]
        return random.randint(lower, upper)

    while time.time() - start_time < time_limit - 5:
        if T < 0.1:
            # Reheat slightly if stuck? Or just stop?
            # Let's keep cooling, it will converge.
            pass
            
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        old_val = current_solution[row_idx][pos]
        new_val = get_new_val(current_solution[row_idx], pos)
        
        if new_val is None or new_val == old_val:
            continue
            
        # Create new row
        new_row = list(current_solution[row_idx])
        new_row[pos] = new_val
        
        # Check internal validity
        if any(new_row[i] >= new_row[i+1] for i in range(k)):
            continue
            
        new_row_diffs = get_row_diffs(new_row)
        if len(new_row_diffs) != k*(k+1)//2:
            continue
            
        # Check cross-row conflicts
        conflict = False
        for i in range(n_rows):
            if i != row_idx:
                if new_row_diffs & current_diffs[i]:
                    conflict = True
                    break
                
        if conflict:
            continue
            
        # Calculate new max
        # Only this row changed
        new_row_max = max(new_row)
        other_max = max(max(current_solution[i]) for i in range(n_rows) if i != row_idx) if n_rows > 1 else 0
        new_max = max(new_row_max, other_max)
        
        delta = new_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_solution[row_idx] = new_row
            current_diffs[row_idx] = new_row_diffs
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_solution = [list(r) for r in current_solution]
                best_diffs = [set(d) for d in current_diffs]
                
        T *= cooling_rate
        
    return best_solution
