"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    for step in range(1, k + 1):
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows

def _validate_rows(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _tighten_polish(rows, n_rows, k, time_limit=50.0):
    """
    Deterministic tightening: repeatedly try to reduce each element to its smallest valid value.
    This is the most effective method for reducing scope in difference triangle sets.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    # Build diff counts array
    max_val = max(max(r) for r in current_rows)
    diff_counts = [0] * (max_val + 20)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                if lower_bound >= old_val:
                    continue
                
                # Try to find the smallest valid value >= lower_bound
                for new_val in range(lower_bound, old_val):
                    # Compute old and new differences
                    old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                    new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                    
                    # Temporarily remove old differences
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] -= 1
                    
                    # Check if new differences are valid
                    valid = True
                    for d in new_diffs:
                        if d >= len(diff_counts):
                            diff_counts.extend([0] * (d - len(diff_counts) + 1))
                        if diff_counts[d] > 0:
                            valid = False
                            break
                    
                    if valid:
                        # Apply move
                        row[pos] = new_val
                        for d in new_diffs:
                            diff_counts[d] += 1
                        improved = True
                        break
                    else:
                        # Restore old differences
                        for d in old_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
    
    return current_rows

def _stochastic_polish(rows, n_rows, k, time_limit=30.0, seed=42):
    """
    Stochastic local search with simulated annealing to escape local minima.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    # Build diff counts
    max_val = max(max(r) for r in current_rows)
    diff_counts = [0] * (max_val + 20)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    
    # Annealing parameters
    T = 5.0
    T_min = 0.1
    alpha = 0.995
    
    while (time.time() - start_time) < time_limit:
        if T <= T_min:
            break
            
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        row = current_rows[r]
        old_val = row[pos]
        
        # Generate random perturbation
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Ensure validity of range
        lower_bound = row[pos - 1] + 1
        upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 10
        
        if new_val < lower_bound:
            new_val = lower_bound
        if new_val > upper_bound:
            new_val = upper_bound
            
        if new_val == old_val:
            continue
            
        # Compute old and new differences
        old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
        new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
        
        # Temporarily remove old differences
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] -= 1
        
        # Check validity
        valid = True
        for d in new_diffs:
            if d >= len(diff_counts):
                diff_counts.extend([0] * (d - len(diff_counts) + 1))
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old differences
            for d in old_diffs:
                if d < len(diff_counts):
                    diff_counts[d] += 1
            continue
        
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
        
        new_scope = get_scope()
        
        if new_scope < current_scope:
            # Accept improvement
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Consider worsening move
            delta_e = new_scope - current_scope
            if random.random() < pow(2.71828, -delta_e / T):
                current_scope = new_scope
            else:
                # Reject, revert
                row[pos] = old_val
                for d in new_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                for d in old_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] += 1
                    
        T *= alpha
        
    return best_rows

def _multi_seed_greedy(n_rows, k, num_seeds=20, time_limit=10.0):
    """Generate multiple greedy solutions with different seeds and return the best."""
    start_time = time.time()
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit:
            break
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Phase 1: Generate diverse initial solutions
    best_rows, best_scope = _multi_seed_greedy(n_rows, k, num_seeds=30, time_limit=15.0)
    
    if best_rows is None:
        # Fallback
        rows = []
        used = set()
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
    
    # Phase 2: Deterministic tightening (most important)
    time_remaining = deadline - time.time()
    t_tighten = min(time_remaining * 0.5, 60.0)
    if t_tighten > 5.0:
        tightened = _tighten_polish(best_rows, n_rows, k, time_limit=t_tighten)
        sc_tighten = max(max(r) for r in tightened)
        if sc_tighten < best_scope:
            best_scope = sc_tighten
            best_rows = tightened
    
    # Phase 3: Stochastic polish to escape local minima
    time_remaining = deadline - time.time()
    t_stoch = min(time_remaining * 0.4, 40.0)
    if t_stoch > 5.0:
        polished = _stochastic_polish(best_rows, n_rows, k, time_limit=t_stoch, seed=123)
        sc_stoch = max(max(r) for r in polished)
        if sc_stoch < best_scope:
            best_scope = sc_stoch
            best_rows = polished
    
    # Phase 4: Final deterministic tighten pass
    time_remaining = deadline - time.time()
    t_final = min(time_remaining * 0.5, 20.0)
    if t_final > 2.0:
        final_rows = _tighten_polish(best_rows, n_rows, k, time_limit=t_final)
        sc_final = max(max(r) for r in final_rows)
        if sc_final < best_scope:
            best_scope = sc_final
            best_rows = final_rows
    
    return best_rows
# EVOLVE-BLOCK-END
