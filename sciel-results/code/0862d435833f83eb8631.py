"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

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
    Uses adaptive step sizes and a more aggressive cooling schedule.
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
    
    # Adaptive annealing parameters
    T = 8.0
    T_min = 0.05
    alpha = 0.998
    no_improve_count = 0
    step_multiplier = 1.0
    
    while (time.time() - start_time) < time_limit:
        if T <= T_min:
            break
            
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        row = current_rows[r]
        old_val = row[pos]
        
        # Adaptive step size: larger when stuck, smaller when improving
        base_delta = int(5 * step_multiplier)
        if base_delta < 1:
            base_delta = 1
        delta = random.randint(-base_delta, base_delta)
        new_val = old_val + delta
        
        # Ensure validity of range
        lower_bound = row[pos - 1] + 1
        upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 15
        
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
            no_improve_count = 0
            # Shrink step size slightly on improvement
            step_multiplier = max(0.5, step_multiplier * 0.95)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Consider worsening move
            delta_e = new_scope - current_scope
            if random.random() < math.exp(-delta_e / T):
                current_scope = new_scope
                no_improve_count += 1
            else:
                # Reject, revert
                row[pos] = old_val
                for d in new_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                for d in old_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] += 1
                no_improve_count += 1
        
        # Increase step size if stuck for a while
        if no_improve_count > 100:
            step_multiplier = min(3.0, step_multiplier * 1.2)
            no_improve_count = 0
            
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

def _row_replacement_sa(rows, n_rows, k, time_limit=20.0, seed=99):
    """
    Simulated Annealing that replaces entire rows.
    This helps escape local minima where single-element moves are trapped.
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
    
    T = 5.0
    T_min = 0.1
    alpha = 0.995
    
    while (time.time() - start_time) < time_limit:
        if T <= T_min:
            break
            
        r = random.randint(0, n_rows - 1)
        
        # Remove old row
        old_row = current_rows[r]
        old_diffs = []
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.append(old_row[j] - old_row[i])
        
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] -= 1
        
        # Generate new row using greedy with current used diffs
        new_row = [0]
        x = 0
        while len(new_row) < k + 1:
            x += 1
            new_diffs = {x - a for a in new_row}
            # Check against current diff_counts (excluding the row we're replacing)
            valid = True
            for d in new_diffs:
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                if diff_counts[d] > 0:
                    valid = False
                    break
            if valid:
                new_row.append(x)
                for d in new_diffs:
                    diff_counts[d] += 1
        
        # Calculate new scope
        new_scope = get_scope()
        
        if new_scope < current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            delta_e = new_scope - current_scope
            if random.random() < math.exp(-delta_e / T):
                current_scope = new_scope
            else:
                # Revert
                # Remove new row diffs
                new_diffs = []
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.append(new_row[j] - new_row[i])
                for d in new_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                # Restore old row diffs
                for d in old_diffs:
                    diff_counts[d] += 1
                current_rows[r] = old_row
        
        T *= alpha
        
    return best_rows

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
    t_tighten = min(time_remaining * 0.4, 50.0)
    if t_tighten > 5.0:
        tightened = _tighten_polish(best_rows, n_rows, k, time_limit=t_tighten)
        sc_tighten = max(max(r) for r in tightened)
        if sc_tighten < best_scope:
            best_scope = sc_tighten
            best_rows = tightened
    
    # Phase 3: Stochastic polish (adaptive SA)
    time_remaining = deadline - time.time()
    t_stoch = min(time_remaining * 0.3, 35.0)
    if t_stoch > 5.0:
        polished = _stochastic_polish(best_rows, n_rows, k, time_limit=t_stoch, seed=123)
        sc_stoch = max(max(r) for r in polished)
        if sc_stoch < best_scope:
            best_scope = sc_stoch
            best_rows = polished
    
    # Phase 4: Row replacement SA (inspiration from other solutions)
    time_remaining = deadline - time.time()
    t_replace = min(time_remaining * 0.2, 20.0)
    if t_replace > 5.0:
        replaced = _row_replacement_sa(best_rows, n_rows, k, time_limit=t_replace, seed=99)
        sc_replace = max(max(r) for r in replaced)
        if sc_replace < best_scope:
            best_scope = sc_replace
            best_rows = replaced
    
    # Phase 5: Final deterministic tighten pass
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
