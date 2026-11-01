"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (the scope).
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Best known solution for (7,5) from Shehadeh et al. 2025: largest entry 112
    # We'll try to find good solutions using a combination of strategies
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Greedy with randomization and backtracking
    def greedy_with_backtrack(n_rows, k, rng, max_backtracks=50):
        used = set()
        rows = []
        
        def build_row(current_row, used_set, max_val):
            if len(current_row) == k + 1:
                return True
            
            start = current_row[-1] + 1 if current_row else 1
            # Try values in randomized order to escape local minima
            candidates = list(range(start, max_val + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_set:
                    continue
                
                used_set.update(new_diffs)
                current_row.append(x)
                
                if build_row(current_row, used_set, max_val):
                    return True
                
                current_row.pop()
                used_set.difference_update(new_diffs)
                
                if time.time() > deadline:
                    return False
            
            return False
        
        # Start with a reasonable upper bound
        # Simple counting bound: need total_diffs distinct values from 1..m
        # So m(m+1)/2 >= total_diffs, m >= ~sqrt(2*total_diffs)
        lower_bound = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
        
        # Try increasing max_val
        for max_val in range(lower_bound, lower_bound + 30):
            if time.time() > deadline:
                break
            
            used_set = set()
            current_row = [0]
            
            if build_row(current_row, used_set, max_val):
                rows.append(current_row)
                # Continue building remaining rows
                success = True
                for _ in range(n_rows - 1):
                    if time.time() > deadline:
                        success = False
                        break
                    row = [0]
                    if not build_row(row, used_set, max_val):
                        success = False
                        break
                    rows.append(row)
                
                if success:
                    scope = max(max(r) for r in rows)
                    return rows, scope
        
        return None, float('inf')
    
    # Strategy 2: Iterated local search
    def local_search(n_rows, k, rng, iters=200):
        # Start with greedy solution
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
        
        scope = max(max(r) for r in rows)
        
        for it in range(iters):
            if time.time() > deadline:
                break
            
            # Pick a random row and try to reduce its max
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            old_max = row[-1]
            
            # Try to find a better arrangement for this row
            # Remove this row's differences from used set
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            for d in row_diffs:
                used.discard(d)
            
            # Try to rebuild this row with smaller max
            best_new_row = None
            for target_max in range(old_max - 1, -1, -1):
                if target_max <= 0:
                    break
                
                def try_build(current, used_set, target):
                    if len(current) == k + 1:
                        return True
                    start = current[-1] + 1 if current else 1
                    for x in range(start, target + 1):
                        new_d = {x - a for a in current}
                        if new_d & used_set:
                            continue
                        used_set.update(new_d)
                        current.append(x)
                        if try_build(current, used_set, target):
                            return True
                        current.pop()
                        used_set.difference_update(new_d)
                    return False
                
                if try_build([0], used, target_max):
                    # Rebuild the row properly
                    row = [0]
                    x = 0
                    # We need to actually extract the row from the build process
                    # Let's do it differently
                    break
                
                if time.time() > deadline:
                    break
            
            # If we didn't find improvement, restore
            for d in row_diffs:
                used.add(d)
        
        return rows, scope
    
    # Strategy 3: Simulated annealing style
    def simulated_annealing(n_rows, k, rng, iters=500):
        # Initial solution via greedy
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
        
        current_scope = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        T = 10.0
        cooling = 0.995
        
        for it in range(iters):
            if time.time() > deadline:
                break
            
            # Pick a random row and perturb it
            row_idx = rng.randint(0, n_rows - 1)
            old_row = rows[row_idx][:]
            
            # Compute old diffs
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            # Remove from used
            for d in old_diffs:
                used.discard(d)
            
            # Try a random modification
            new_row = old_row[:]
            # Randomly adjust one or two positions
            for _ in range(rng.randint(1, 2)):
                pos = rng.randint(1, len(new_row) - 1)  # Don't change 0
                delta = rng.randint(-5, 5)
                new_val = new_row[pos] + delta
                if new_val <= new_row[pos - 1] or new_val >= new_row[pos + 1] if pos < len(new_row) - 1 else new_val <= new_row[pos - 1]:
                    continue
                if new_val <= 0:
                    continue
                new_row[pos] = new_val
            
            # Check validity
            new_diffs = set()
            valid = True
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in new_diffs or d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                new_scope = max(max(r) for r in rows) if row_idx != n_rows - 1 else max(new_row[-1], max(max(r) for i, r in enumerate(rows) if i != row_idx))
                # Simpler: compute full scope
                new_scope = max(max(r) for r in rows)
                # Update rows temporarily
                rows[row_idx] = new_row
                for d in new_diffs:
                    used.add(d)
                
                # Accept if better or with probability
                if new_scope <= current_scope:
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                else:
                    if rng.random() < math.exp(-(new_scope - current_scope) / T):
                        current_scope = new_scope
                    else:
                        # Reject
                        rows[row_idx] = old_row
                        for d in new_diffs:
                            used.discard(d)
                        for d in old_diffs:
                            used.add(d)
            else:
                # Restore
                for d in old_diffs:
                    used.add(d)
            
            T *= cooling
        
        return best_rows, best_scope
    
    # Run multiple strategies with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Try greedy with backtracking
        rows, scope = greedy_with_backtrack(n_rows, k, rng)
        if rows is not None:
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
        
        # Try simulated annealing
        rows, scope = simulated_annealing(n_rows, k, rng, iters=200)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # If we haven't found anything, fall back to simple greedy
    if best_rows is None:
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
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
