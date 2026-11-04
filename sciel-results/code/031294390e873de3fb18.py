"""Program for Difference Triangle Set (4,7) - bitmask-based local search."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # For our specific case: n_rows=4, k=7 -> diffs_per_row=28, total=112
    # Best known: 134
    
    def get_scope(rows):
        return max(max(row) for row in rows) if rows else 0
    
    def compute_all_diffs(rows):
        """Return list of all differences (with row index)."""
        diffs = []
        for r_idx, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diffs.append((d, r_idx, i, j))
        return diffs
    
    def count_conflicts(diffs):
        """Count number of distinct differences that appear more than once."""
        diff_counts = {}
        for d, r_idx, i, j in diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = sum(1 for c in diff_counts.values() if c > 1)
        return conflicts, diff_counts
    
    def greedy_construct(max_val):
        """Greedily construct rows, picking smallest valid next integer."""
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            for _pos in range(k):
                found = False
                next_val = current_max + 1
                while next_val <= max_val:
                    valid = True
                    for prev in row:
                        diff = next_val - prev
                        if diff in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(next_val)
                        current_max = next_val
                        for prev in row[:-1]:
                            used_diffs.add(next_val - prev)
                        found = True
                        break
                    next_val += 1
                if not found:
                    return None
            rows.append(row)
        return rows
    
    def random_rows(max_val):
        """Generate random rows within max_val."""
        rows = []
        for _ in range(n_rows):
            # Generate k+1 increasing integers starting at 0, ending <= max_val
            # Use random gaps
            row = [0]
            remaining = max_val
            # Generate k gaps that sum to <= max_val
            # Simple approach: pick random positions
            for _ in range(k):
                if not row:
                    break
                # Pick next value between last+1 and max_val
                # Bias towards smaller to allow room
                low = row[-1] + 1
                high = max_val
                if low > high:
                    break
                # Weight towards smaller values
                new_val = random.randint(low, high)
                row.append(new_val)
            # Ensure exactly k+1 elements
            while len(row) < k + 1:
                row.append(row[-1] + 1)
            rows.append(row[:k+1])
        return rows
    
    def local_search(rows, max_val, time_limit, T_init=30.0, cooling=0.999):
        """Simulated annealing to minimize conflicts and scope."""
        rows = [row[:] for row in rows]
        T = T_init
        
        # Compute initial state
        diffs = compute_all_diffs(rows)
        conf_count, diff_counts = count_conflicts(diffs)
        current_scope = get_scope(rows)
        current_cost = conf_count * 10000 + max(0, current_scope - max_val) * 100
        
        best_rows = [row[:] for row in rows]
        best_cost = current_cost
        best_scope = current_scope
        
        while time.time() < time_limit:
            # Pick random row and position
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            
            row = rows[r_idx]
            min_val = row[c_idx - 1] + 1
            max_val_limit = max_val
            if c_idx < k:
                max_val_limit = min(max_val_limit, row[c_idx + 1] - 1)
            
            if min_val > max_val_limit:
                T *= cooling
                continue
            
            # Try random new value
            new_val = random.randint(min_val, max_val_limit)
            if new_val == row[c_idx]:
                T *= cooling
                continue
            
            old_val = row[c_idx]
            row[c_idx] = new_val
            
            # Compute new cost
            diffs = compute_all_diffs(rows)
            conf_count, diff_counts = count_conflicts(diffs)
            new_scope = get_scope(rows)
            new_cost = conf_count * 10000 + max(0, new_scope - max_val) * 100
            
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                current_scope = new_scope
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                if conf_count == 0:
                    # Valid solution!
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
            else:
                row[c_idx] = old_val
            
            T *= cooling
            if T < 0.1:
                break
        
        return best_rows, best_scope, best_cost
    
    # Phase 1: Find initial valid solution
    best_rows = None
    best_scope = float('inf')
    
    # Try greedy with increasing max_val
    current_max = lower_bound
    found_initial = False
    
    while time.time() < deadline - 10.0 and not found_initial:
        rows = greedy_construct(current_max)
        if rows:
            best_rows = rows
            best_scope = get_scope(rows)
            found_initial = True
            break
        current_max = int(current_max * 1.15) + 5
        if current_max > 5000:
            break
    
    if not found_initial:
        # Fallback: large max_val
        rows = greedy_construct(5000)
        if rows:
            return rows
        # Last resort
        rows = []
        base = 1
        for _ in range(n_rows):
            row = [0]
            for i in range(1, k+1):
                row.append(base * (i+1))
            rows.append(row)
            base *= (k+2)
        return rows
    
    # Phase 2: Improve using local search
    # Try to reduce scope
    target_scope = best_scope - 1
    
    for attempt in range(30):
        if time.time() > deadline - 5.0:
            break
        
        # Time budget for this attempt
        time_left = deadline - time.time() - 5.0
        if time_left < 1.0:
            break
        
        # Try local search with target scope
        T_limit = min(8.0, time_left * 0.3)
        
        # Start from best_rows
        improved_rows, improved_scope, _ = local_search(
            best_rows, target_scope, T_limit
        )
        
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
            target_scope = best_scope - 1
    
    # Phase 3: Multi-start local search from random initializations
    time_left = deadline - time.time() - 5.0
    if time_left > 5.0:
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            time_left = deadline - time.time() - 5.0
            if time_left < 2.0:
                break
            
            # Random initialization within current best scope
            T_limit = min(5.0, time_left * 0.25)
            
            # Generate random rows
            rand_rows = random_rows(best_scope)
            
            improved_rows, improved_scope, _ = local_search(
                rand_rows, best_scope - 1, T_limit
            )
            
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows
    
    return best_rows

# EVOLVE-BLOCK-END
