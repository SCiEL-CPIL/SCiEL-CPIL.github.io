import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solution with greedy
    seed = 0
    while time.time() < deadline - 60:  # Reserve 60s for SA
        rows = _greedy_construct(n_rows, k, seed)
        scope = _scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope == lower_bound:
                break
        
        seed += 1
    
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated annealing to improve
    if time.time() < deadline - 5:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            last_x = row[-1]
            limit = 500
            
            x = last_x + 1
            while len(valid_xs) < 16 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 16:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            chosen_x = rng.choice(valid_xs)
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """Simulated annealing with coordinated row moves."""
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Temperature schedule
    T_start = 50.0
    T_end = 1.0
    total_steps = 20000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Choose a random row to modify
        row_idx = rng.randint(0, n_rows - 1)
        
        # Save old row
        old_row = list(rows[row_idx])
        
        # Generate a new candidate row using greedy with some randomness
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        
        if new_row is None:
            continue
        
        # Compute old and new scopes
        old_scope = current_scope
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        # Check validity: all differences must be distinct
        all_diffs = _all_diffs(new_rows)
        expected_diffs = n_rows * (k + 1) * k // 2
        if len(all_diffs) != expected_diffs:
            # Invalid, revert
            continue
        
        # Accept or reject based on Metropolis criterion
        delta = new_scope - old_scope
        if delta <= 0:
            # Improvement, always accept
            rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Worse, accept with probability exp(-delta/T)
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                rows = new_rows
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows

def _generate_candidate_row(rows, exclude_idx, k, rng):
    """Generate a new candidate row that avoids conflicts with other rows."""
    # Collect all differences used by other rows
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a+1, n):
                used_diffs.add(row[b] - row[a])
    
    # Build new row greedily, avoiding used_diffs
    row = [0]
    for _ in range(k):
        valid_xs = []
        last_x = row[-1]
        limit = 500
        
        x = last_x + 1
        while len(valid_xs) < 20 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            # Try harder
            x = last_x + 1
            while len(valid_xs) < 20:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                x += 1
        
        if not valid_xs:
            return None
        
        # Pick with some bias toward smaller values
        if rng.random() < 0.3:
            # Random choice
            chosen_x = rng.choice(valid_xs)
        else:
            # Prefer smaller
            chosen_x = valid_xs[rng.randint(0, min(3, len(valid_xs)-1))]
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
    
    return row
