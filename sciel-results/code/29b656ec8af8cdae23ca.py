import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit (150s kill)
    
    # Total number of distinct differences needed
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs  # Simple counting bound

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            # Check strictly increasing
            for i in range(1, k + 1):
                if r[i] <= r[i-1]:
                    return False
            # Check distinct differences
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def compute_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct(M_limit, seed_val=0):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                next_val = row[-1] + 1
                valid_found = False
                while next_val <= M_limit:
                    new_diffs = set()
                    valid = True
                    for prev in row:
                        d = next_val - prev
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Check internal consistency
                        curr_diffs = compute_row_diffs(row)
                        if new_diffs & curr_diffs:
                            valid = False
                        
                        if valid:
                            row.append(next_val)
                            used_diffs.update(new_diffs)
                            valid_found = True
                            break
                    next_val += 1
                
                if not valid_found:
                    return None
            rows.append(row)
        return rows

    def local_search_polish(rows, time_budget, seed_val=0):
        """
        Systematically minimize each element by trying to decrease it.
        This is a deterministic local search that helps escape shallow local minima.
        """
        if rows is None:
            return None, None
            
        rng = random.Random(seed_val)
        current = [r[:] for r in rows]
        current_scope = get_scope(current)
        best = [r[:] for r in current]
        best_scope = current_scope
        
        used_diffs = get_used_diffs(current)
        deadline = time.time() + time_budget
        
        # Repeat passes until no improvement or time runs out
        while time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                if time.time() > deadline:
                    break
                row = current[ri]
                old_diffs = compute_row_diffs(row)
                used_without = used_diffs - old_diffs
                
                # Try to decrease each element from right to left
                for idx in range(k, 0, -1):
                    if time.time() > deadline:
                        break
                    
                    old_val = row[idx]
                    lower_bound_val = row[idx-1] + 1
                    upper_bound_val = old_val - 1  # Only try to decrease
                    
                    if lower_bound_val > upper_bound_val:
                        continue
                    
                    # Try values from old_val - 1 down to lower_bound_val
                    # Limit search to avoid excessive time per element
                    max_attempts = 50
                    for new_val in range(old_val - 1, lower_bound_val - 1, -1):
                        if time.time() > deadline:
                            break
                        if max_attempts <= 0:
                            break
                        max_attempts -= 1
                        
                        new_row = row[:]
                        new_row[idx] = new_val
                        new_diffs = compute_row_diffs(new_row)
                        
                        if len(new_diffs) != diffs_per_row:
                            continue
                        
                        if new_diffs & used_without:
                            continue
                        
                        # Valid move
                        other_max = 0
                        for i, r in enumerate(current):
                            if i != ri:
                                other_max = max(other_max, max(r))
                        new_scope = max(other_max, max(new_row))
                        
                        if new_scope < current_scope:
                            current[ri] = new_row
                            used_diffs = used_without | new_diffs
                            current_scope = new_scope
                            improved = True
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best = [r[:] for r in current]
                            break  # Restart for this row since structure changed
                        elif new_scope == current_scope:
                            # Accept neutral moves to explore
                            if rng.random() < 0.1:
                                current[ri] = new_row
                                used_diffs = used_without | new_diffs
                                improved = True
                            break
            
            if not improved:
                break
        
        return best, best_scope

    def simulated_annealing(rows, time_budget, seed_val=0):
        if rows is None:
            return None, None
            
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used_diffs = get_used_diffs(current)
        
        T = 10.0
        cooling = 0.995
        deadline = time.time() + time_budget
        
        step = 0
        while time.time() < deadline and T > 0.1:
            step += 1
            
            # Pick a random row
            ri = rng.randint(0, n_rows - 1)
            row = current[ri]
            old_diffs = compute_row_diffs(row)
            used_without = used_diffs - old_diffs
            
            # Pick a random element index to change (1 to k)
            if rng.random() < 0.5:
                # Target the last element to reduce scope
                idx = k
            else:
                idx = rng.randint(1, k)
            
            old_val = row[idx]
            
            # Determine range for new value
            lower_bound_val = row[idx-1] + 1
            upper_bound_val = row[idx+1] - 1 if idx < k else current_scope
            
            # If we are trying to reduce scope, bias towards lower values
            new_val = None
            # Try a few random candidates in the valid range
            for _ in range(50):
                if lower_bound_val > upper_bound_val:
                    break
                # Sample from range
                if rng.random() < 0.3:
                    if old_val > lower_bound_val:
                        new_val = rng.randint(lower_bound_val, min(upper_bound_val, old_val - 1))
                    else:
                        new_val = None
                else:
                    new_val = rng.randint(lower_bound_val, upper_bound_val)
                
                if new_val is None or new_val == old_val:
                    continue
                    
                # Check validity
                new_row = row[:]
                new_row[idx] = new_val
                new_diffs = compute_row_diffs(new_row)
                
                if new_diffs & used_without:
                    continue
                
                if len(new_diffs) != diffs_per_row:
                    continue
                
                # Valid move
                other_max = 0
                for i, r in enumerate(current):
                    if i != ri:
                        other_max = max(other_max, max(r))
                new_scope = max(other_max, max(new_row))
                
                delta = new_scope - current_scope
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    # Accept
                    current[ri] = new_row
                    used_diffs = used_without | new_diffs
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                    break
            
            T *= cooling
            
        return best, best_scope

    def row_replacement_search(rows, time_budget, seed_val=0):
        """
        Escape local minima by rebuilding entire rows.
        """
        if rows is None:
            return None, None
            
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used_diffs = get_used_diffs(current)
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            # Pick a random row to replace
            ri = rng.randint(0, n_rows - 1)
            old_diffs = compute_row_diffs(current[ri])
            used_without = used_diffs - old_diffs
            
            # Try to build a new row with smaller scope
            new_row = None
            # Try multiple seeds for diversity
            for seed in range(10):
                if time.time() > deadline:
                    break
                
                # Build row greedily with the current used_diffs (excluding old row's diffs)
                row = [0]
                valid_found = True
                while len(row) < k + 1:
                    next_val = row[-1] + 1
                    found = False
                    while next_val <= current_scope:
                        new_diffs = set()
                        valid = True
                        for prev in row:
                            d = next_val - prev
                            if d in used_without:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if valid:
                            curr_diffs = compute_row_diffs(row)
                            if new_diffs & curr_diffs:
                                valid = False
                            else:
                                row.append(next_val)
                                found = True
                                break
                        next_val += 1
                    
                    if not found:
                        valid_found = False
                        break
                
                if valid_found:
                    new_row = row
                    new_diffs = compute_row_diffs(new_row)
                    
                    # Check if this improves scope
                    other_max = 0
                    for i, r in enumerate(current):
                        if i != ri:
                            other_max = max(other_max, max(r))
                    new_scope = max(other_max, max(new_row))
                    
                    if new_scope < current_scope:
                        current[ri] = new_row
                        used_diffs = used_without | new_diffs
                        current_scope = new_scope
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best = [r[:] for r in current]
                        break  # Found improvement, restart
            
        return best, best_scope

    # --- Main Logic ---
    
    # Phase 1: Get a baseline with greedy
    best_rows = None
    best_M = float('inf')
    
    start_M = lower_bound
    end_M = lower_bound + 40
    
    for M in range(start_M, end_M):
        if time.time() > deadline - 30:
            break
        for seed in range(5):
            if time.time() > deadline - 25:
                break
            rows = greedy_construct(M, seed)
            if rows is not None:
                if is_valid(rows):
                    current_max = get_scope(rows)
                    if current_max < best_M:
                        best_M = current_max
                        best_rows = [r[:] for r in rows]
                    break
        if best_rows is not None and best_M <= M:
            break
            
    # Fallback if greedy failed
    if best_rows is None:
        M_huge = 300
        rows = greedy_construct(M_huge, 0)
        if rows:
            best_rows = rows
            best_M = get_scope(rows)
        else:
            best_rows = []
            base = 0
            for i in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    base += 100
                    row.append(base)
                best_rows.append(row)
            best_M = base

    # Phase 2: Local search polish
    time_left = deadline - time.time()
    if time_left > 5:
        ls_budget = min(time_left - 5, 30)
        if ls_budget > 0:
            improved_rows, improved_M = local_search_polish(best_rows, ls_budget, seed_val=42)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows

    # Phase 3: Simulated annealing
    time_left = deadline - time.time()
    if time_left > 5:
        sa_budget = min(time_left - 5, 40)
        if sa_budget > 0:
            improved_rows, improved_M = simulated_annealing(best_rows, sa_budget, seed_val=123)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows

    # Phase 4: Row replacement search
    time_left = deadline - time.time()
    if time_left > 5:
        rr_budget = min(time_left - 5, 30)
        if rr_budget > 0:
            improved_rows, improved_M = row_replacement_search(best_rows, rr_budget, seed_val=456)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows

    # Phase 5: Final polish
    time_left = deadline - time.time()
    if time_left > 3:
        final_budget = min(time_left - 3, 10)
        if final_budget > 0:
            improved_rows, improved_M = local_search_polish(best_rows, final_budget, seed_val=789)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows

    return best_rows

if __name__ == "__main__":
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", get_scope(rows))
        print("Valid:", is_valid(rows))
    else:
        print("No solution found")
