import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    max_time = 115.0
    
    best_rows = None
    best_energy = float('inf')
    
    def calculate_energy(rows):
        used_diffs = {}
        conflicts = 0
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diff = r[j] - r[i]
                    if diff in used_diffs:
                        conflicts += 1
                    else:
                        used_diffs[diff] = 1
        return conflicts * 1000 + max_val

    def is_valid(rows):
        used_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diff = r[j] - r[i]
                    if diff in used_diffs:
                        return False
                    used_diffs.add(diff)
        return True

    def get_scope(rows):
        return max(r[-1] for r in rows)

    def random_row(max_val):
        # Generate a strictly increasing row starting with 0
        # Length k+1
        row = [0]
        curr = 0
        for _ in range(k):
            # Pick next value > curr
            # Bias towards small values
            # Upper bound for this step to keep row compact
            # Let's pick from curr+1 to min(max_val, curr + 10)
            upper = min(max_val, curr + 10)
            if upper <= curr:
                upper = curr + 1
            next_val = random.randint(curr + 1, upper)
            row.append(next_val)
            curr = next_val
        return row

    # Initialize with several random restarts
    # We will run SA for a short duration, then restart if no improvement.
    
    for restart in range(50):
        if time.time() - start_time > max_time:
            break
            
        # Initialize state
        # Start with a "tight" range to encourage small scope
        init_max = 50
        rows = [random_row(init_max) for _ in range(n_rows)]
        
        current_energy = calculate_energy(rows)
        best_rows_in_run = [r[:] for r in rows]
        best_energy_in_run = current_energy
        
        T = 100.0
        cooling = 0.995
        
        while time.time() - start_time < max_time:
            # Generate neighbor
            # Pick random row and position
            r_idx = random.randint(0, n_rows - 1)
            p_idx = random.randint(1, k) # 0 is fixed
            
            row = rows[r_idx]
            curr_val = row[p_idx]
            
            # Determine valid range for new value
            low = row[p_idx - 1] + 1
            if p_idx < k:
                high = row[p_idx + 1] - 1
            else:
                # For the last element, we can go up to a reasonable limit
                # To minimize scope, we should try to keep it low
                # But we also need to allow increases if we are stuck
                high = max(curr_val + 10, 100) 
                
            if low > high:
                # Cannot move, try another position or skip
                continue
                
            # Bias towards smaller values to reduce scope
            # Pick a value in [low, high]
            # Weighted random choice: prefer low
            # Simple: pick from [low, min(high, low + 5)] with prob 0.8, else [low, high]
            if random.random() < 0.8:
                search_high = min(high, low + 5)
                if search_high < low:
                    search_high = low
                new_val = random.randint(low, search_high)
            else:
                new_val = random.randint(low, high)
                
            if new_val == curr_val:
                continue
                
            # Apply move
            old_val = row[p_idx]
            row[p_idx] = new_val
            
            new_energy = calculate_energy(rows)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_energy = new_energy
                if current_energy < best_energy_in_run:
                    best_energy_in_run = current_energy
                    best_rows_in_run = [r[:] for r in rows]
            else:
                row[p_idx] = old_val
                
            T *= cooling
            if T < 0.1:
                break
        
        # After SA run, check if we found a valid solution
        if best_energy_in_run < best_energy:
            # Check if valid
            if is_valid(best_rows_in_run):
                best_energy = best_energy_in_run
                best_rows = best_rows_in_run
                # If we found a valid solution, we can continue to optimize scope
                # Or just restart to find a better one
                # For now, update global best
            else:
                # Not valid, but might be a good starting point for next restart?
                # We only track valid best_rows for the final answer
                # But we can use the "best energy" to guide?
                # The task requires a VALID result.
                pass
        
        # If we found a valid solution, we might want to do a "scope minimization" phase
        # But SA with energy = conflicts*1000 + scope handles this.
        # If conflicts=0, it minimizes scope.
        
    if best_rows is None:
        # Fallback: Greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    return best_rows
