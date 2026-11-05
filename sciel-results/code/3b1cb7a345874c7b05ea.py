import random
import math
import time

def construct(n_rows, k):
    # 1. Initial Solution: Greedy
    def generate_greedy():
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
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = None
    best_max = float('inf')
    
    # Helper to calculate cost
    def get_cost(rows):
        m = max(max(r) for r in rows)
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    all_diffs.append(r[j] - r[i])
        
        seen = set()
        conflicts = 0
        for d in all_diffs:
            if d in seen:
                conflicts += 1
            else:
                seen.add(d)
        
        # Cost: Minimize m. If invalid, heavily penalize conflicts.
        # We want to find valid solutions with small m.
        # If we find a valid solution, we want to reduce m.
        # If we have conflicts, we want to resolve them.
        # A good heuristic: cost = m + conflicts * (m + 1)
        # This ensures that resolving a conflict is more important than reducing m slightly,
        # but reducing m is important once valid.
        # Actually, if conflicts > 0, we should prioritize reducing conflicts.
        # Let's use: if conflicts > 0: cost = conflicts * 10000 + m
        # else: cost = m
        # This forces validity first.
        if conflicts > 0:
            return conflicts * 10000 + m, m, conflicts
        else:
            return m, m, conflicts

    # SA Parameters
    # We run multiple independent SA runs with restarts to escape local minima.
    total_time_limit = 110
    start_global_time = time.time()
    
    # Number of restarts
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_global_time > total_time_limit:
            break
            
        rows = generate_greedy()
        current_cost, cur_m, cur_conf = get_cost(rows)
        
        # If greedy is already very good, try to improve
        # T starts high to allow exploration
        T = 50.0
        cooling = 0.995
        min_T = 0.1
        iterations = 5000
        
        sa_start_time = time.time()
        
        while T > min_T:
            if time.time() - start_global_time > total_time_limit:
                break
            if time.time() - sa_start_time > 5: # Limit time per SA run
                break
                
            # Generate Move
            r_idx = random.randint(0, n_rows - 1)
            # Choose move type: shift suffix or change single element
            if random.random() < 0.3:
                # Shift suffix
                if k == 0:
                    continue
                j_idx = random.randint(1, k)
                # Larger steps to escape local minima
                delta = random.randint(-5, 5)
                if delta == 0:
                    continue
                # Check boundary
                if j_idx > 0:
                    if rows[r_idx][j_idx-1] >= rows[r_idx][j_idx] + delta:
                        continue
                # Apply shift
                old_suffix = rows[r_idx][j_idx:]
                new_suffix = [v + delta for v in old_suffix]
                rows[r_idx][j_idx:] = new_suffix
                move_type = 'shift'
                move_data = (r_idx, j_idx, delta, old_suffix)
            else:
                # Change single element
                j_idx = random.randint(1, k)
                delta = random.randint(-5, 5)
                if delta == 0:
                    continue
                new_val = rows[r_idx][j_idx] + delta
                left = rows[r_idx][j_idx-1]
                right = rows[r_idx][j_idx+1] if j_idx < k else float('inf')
                if left < new_val < right:
                    old_val = rows[r_idx][j_idx]
                    rows[r_idx][j_idx] = new_val
                    move_type = 'single'
                    move_data = (r_idx, j_idx, old_val)
                else:
                    continue # Revert/ignore if invalid move

            new_cost, new_m, new_conf = get_cost(rows)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                if new_conf == 0 and new_m < best_max:
                    best_max = new_m
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                if move_type == 'shift':
                    r, j, d, old_s = move_data
                    rows[r][j:] = old_s
                elif move_type == 'single':
                    r, j, old_v = move_data
                    rows[r][j] = old_v
            
            T *= cooling

    if best_rows is None:
        # Fallback to greedy if SA failed to find valid (shouldn't happen as greedy is valid)
        best_rows = generate_greedy()
        
    return best_rows
