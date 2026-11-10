import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def compute_all_diffs(rows):
        """Return set of all positive differences across all rows."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all positive differences are distinct across all rows."""
        total_diffs = n_rows * (k + 1) * k // 2
        return len(compute_all_diffs(rows)) == total_diffs
    
    def greedy_build(seed, n_rows, k):
        """Greedy construction with randomization for tie-breaking."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                for _ in range(50):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            break
            rows.append(row)
        return rows
    
    def backtracking_build(n_rows, k, max_depth=6, timeout=5.0):
        """
        Use backtracking to construct the first few rows optimally,
        then greedy for the rest. This focuses computational effort
        where constraints are tightest.
        """
        start_bt = time.time()
        best_partial = None
        best_partial_max = float('inf')
        
        def backtrack(row_idx, used, rows, current_max):
            nonlocal best_partial, best_partial_max
            if time.time() - start_bt > timeout:
                return False
            
            if row_idx == n_rows:
                if current_max < best_partial_max:
                    best_partial_max = current_max
                    best_partial = [list(r) for r in rows]
                return True
            
            # Try to build row row_idx
            row = [0]
            x = 0
            
            def extend_row(pos, row_so_far, used_so_far, cur_max):
                nonlocal best_partial, best_partial_max
                if time.time() - start_bt > timeout:
                    return False
                
                if pos == k + 1:
                    # Row complete
                    new_used = used_so_far | {row_so_far[j] - row_so_far[i] for i in range(len(row_so_far)) for j in range(i+1, len(row_so_far))}
                    new_max = max(cur_max, row_so_far[-1])
                    if new_max >= best_partial_max:
                        return False
                    if backtrack(row_idx + 1, new_used, rows + [row_so_far], new_max):
                        return True
                    return False
                
                # Try values for this position
                start_val = row_so_far[-1] + 1 if row_so_far else 1
                # Limit search space
                max_val = best_partial_max if best_partial_max < float('inf') else 200
                
                for val in range(start_val, max_val + 1):
                    new_row = row_so_far + [val]
                    # Check increasing
                    if len(new_row) >= 2 and new_row[-1] <= new_row[-2]:
                        continue
                    
                    # Check new differences don't conflict
                    new_diffs = {val - new_row[i] for i in range(len(new_row) - 1)}
                    if new_diffs & used_so_far:
                        continue
                    # Also check within-row differences
                    within_diffs = {new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row))}
                    if within_diffs & used_so_far:
                        continue
                    
                    new_used = used_so_far | within_diffs
                    new_max = max(cur_max, val)
                    
                    if extend_row(pos + 1, new_row, new_used, new_max):
                        return True
                
                return False
            
            extend_row(1, [0], used, current_max)
            return False
        
        backtrack(0, set(), [], 0)
        
        if best_partial is None:
            return None
        
        # Complete the remaining rows greedily
        used = compute_all_diffs(best_partial)
        rows = list(best_partial)
        for _ in range(len(best_partial), n_rows):
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
    
    def simulated_annealing(rows, rng, max_iters=5000):
        """
        Simulated annealing to minimize the maximum entry.
        Uses a proper cooling schedule and accepts worse solutions
        with decreasing probability to escape local minima.
        """
        current = [list(r) for r in rows]
        current_score = max_entry(current)
        best = [list(r) for r in current]
        best_score = current_score
        
        # Initial temperature
        T = current_score * 0.5
        T_min = 0.01
        cooling_rate = 0.9995
        
        for it in range(max_iters):
            if elapsed() > deadline - 5:
                break
            
            # Generate a neighbor: modify one element
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # positions 1..k
            
            current_val = current[row_idx][pos]
            
            # Try a random new value (biased towards lower values)
            if rng.random() < 0.7:
                # Try to reduce
                delta = rng.randint(1, max(1, current_val // 3))
                new_val = current_val - delta
            else:
                # Try to increase slightly (to escape local minima)
                delta = rng.randint(1, max(1, current_val // 10))
                new_val = current_val + delta
            
            if new_val <= 0:
                continue
            
            # Check increasing constraint
            if pos > 0 and new_val <= current[row_idx][pos - 1]:
                continue
            if pos < k and new_val >= current[row_idx][pos + 1]:
                continue
            
            # Apply change
            old_val = current[row_idx][pos]
            current[row_idx][pos] = new_val
            
            # Check validity
            if is_valid(current):
                new_score = max_entry(current)
                delta_score = new_score - current_score
                
                if new_score < current_score:
                    # Accept improvement
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = [list(r) for r in current]
                else:
                    # Accept worsening with probability exp(-delta/T)
                    if rng.random() < math.exp(-delta_score / max(T, T_min)):
                        current_score = new_score
                    else:
                        # Reject
                        current[row_idx][pos] = old_val
            else:
                # Invalid solution, reject
                current[row_idx][pos] = old_val
            
            # Cool down
            T = max(T * cooling_rate, T_min)
        
        return best
    
    # Phase 1: Get initial solutions from multiple strategies
    best_rows = None
    best_val = float('inf')
    
    # Try backtracking-based construction
    if n_rows <= 8 and k <= 10:
        bt_result = backtracking_build(n_rows, k, max_depth=4, timeout=8.0)
        if bt_result is not None:
            val = max_entry(bt_result)
            if val < best_val:
                best_val = val
                best_rows = bt_result
    
    # Try greedy constructions
    for seed in range(30):
        if elapsed() > deadline - 30:
            break
        rows = greedy_build(seed, n_rows, k)
        val = max_entry(rows)
        if val < best_val:
            best_val = val
            best_rows = rows
    
    if best_rows is None:
        # Fallback to simple greedy
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
    
    # Phase 2: Simulated annealing
    rng = random.Random(42)
    best_rows = simulated_annealing(best_rows, rng, max_iters=3000)
    
    # Phase 3: More annealing if time permits
    if elapsed() < deadline - 15:
        rng2 = random.Random(123)
        best_rows2 = simulated_annealing(best_rows, rng2, max_iters=3000)
        if max_entry(best_rows2) < max_entry(best_rows):
            best_rows = best_rows2
    
    return best_rows
