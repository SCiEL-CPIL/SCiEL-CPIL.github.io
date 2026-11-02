"""Improved difference triangle set (7,5) solver using row-rebuild + SA."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 115
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy starts with different randomizations
    for seed in range(20):
        if time.time() >= deadline - 20:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    # Phase 2: Simulated annealing with row-rebuild moves
    if best_rows is not None:
        best_rows, best_scope = _sa_with_row_rebuild(best_rows, n_rows, k, deadline)
    
    # Phase 3: Additional restarts with SA from different initial solutions
    if time.time() < deadline - 30:
        for seed in range(5):
            if time.time() >= deadline - 20:
                break
            rows = _greedy_construct(n_rows, k, seed + 100)
            if rows is not None:
                rows, scope = _sa_with_row_rebuild(rows, n_rows, k, deadline)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
    
    # Phase 4: Final polish
    if best_rows is not None and time.time() < deadline - 10:
        best_rows, best_scope = _final_polish(best_rows, n_rows, k, deadline)
    
    return best_rows

def _greedy_construct(n_rows, k, seed=0):
    """Greedy construction with randomization."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            # Randomized greedy: try next few values
            found = False
            for offset in range(0, 5):
                trial = x + 1 + offset
                new = {trial - a for a in row}
                if not new & used:
                    row.append(trial)
                    used |= new
                    x = trial
                    found = True
                    break
            if not found:
                x += 1
        rows.append(row)
    
    return rows

def _scope(rows):
    return max(max(row) for row in rows)

def _compute_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _sa_with_row_rebuild(rows, n_rows, k, deadline):
    """SA with both element moves and full row rebuilds."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    rng = random.Random(42)
    start_temp = 80.0
    end_temp = 0.1
    cooling_rate = 0.997
    
    row_diffs = [_compute_diffs(r) for r in current_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    temp = start_temp
    iterations = 0
    max_iterations = 80000
    
    while time.time() < deadline - 12 and temp > end_temp and iterations < max_iterations:
        iterations += 1
        
        # Choose move type: 30% element move, 70% row rebuild
        if rng.random() < 0.3:
            # Element move
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) <= 1:
                temp *= cooling_rate
                continue
            
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            delta = rng.randint(-20, 20)
            if delta == 0:
                temp *= cooling_rate
                continue
            new_val = old_val + delta
            
            if new_val <= 0:
                temp *= cooling_rate
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                temp *= cooling_rate
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                temp *= cooling_rate
                continue
            
            old_diffs = _compute_diffs(row)
            row[pos] = new_val
            new_diffs = _compute_diffs(row)
            
            expected_diff_count = len(row) * (len(row) - 1) // 2
            if len(new_diffs) != expected_diff_count:
                row[pos] = old_val
                temp *= cooling_rate
                continue
            
            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                row[pos] = old_val
                temp *= cooling_rate
                continue
            
            new_scope = _scope(current_rows)
            delta_scope = new_scope - current_scope
            
            if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temp):
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diffs[ri] = new_diffs
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                row[pos] = old_val
        else:
            # Row rebuild: replace entire row with a new valid row
            ri = rng.randrange(n_rows)
            old_row = list(current_rows[ri])
            old_diffs = row_diffs[ri]
            other_diffs = all_diffs - old_diffs
            
            # Try to build a new row avoiding other_diffs
            new_row = _build_row_randomized(rng, k, other_diffs, old_row)
            if new_row is not None:
                new_diffs = _compute_diffs(new_row)
                new_scope = _scope(current_rows[:ri] + [new_row] + current_rows[ri+1:])
                delta_scope = new_scope - current_scope
                
                if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temp):
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    all_diffs = (all_diffs - old_diffs) | new_diffs
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
        
        temp *= cooling_rate
    
    return best_rows, best_scope

def _build_row_randomized(rng, k, forbidden_diffs, old_row):
    """Try to build a new row that avoids forbidden differences."""
    # Start from a randomized version of the old row
    row = [0]
    current_max = 0
    
    for _ in range(k):
        found = False
        # Try values around where the old row had entries, plus random
        candidates = []
        if old_row:
            for v in old_row[1:]:
                candidates.append(v + rng.randint(-5, 5))
        candidates.extend(range(current_max + 1, current_max + 30))
        rng.shuffle(candidates)
        
        for next_val in candidates:
            if next_val <= current_max:
                continue
            new_diffs = {next_val - a for a in row}
            if new_diffs & forbidden_diffs:
                continue
            # Check internal consistency (distinct diffs within row)
            # Since we're building incrementally, just check against existing
            row.append(next_val)
            current_max = next_val
            found = True
            break
        
        if not found:
            return None
    
    return row

def _final_polish(rows, n_rows, k, deadline):
    """Final local polish."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    rng = random.Random(123)
    
    row_diffs = [_compute_diffs(r) for r in best_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    for _ in range(30000):
        if time.time() >= deadline - 5:
            break
        
        ri = rng.randrange(n_rows)
        row = best_rows[ri]
        if len(row) <= 1:
            continue
        
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        for delta in range(-5, 6):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            old_diffs = _compute_diffs(row)
            row[pos] = new_val
            new_diffs = _compute_diffs(row)
            
            expected_diff_count = len(row) * (len(row) - 1) // 2
            if len(new_diffs) != expected_diff_count:
                row[pos] = old_val
                continue
            
            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                row[pos] = old_val
                continue
            
            new_scope = _scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diffs[ri] = new_diffs
                break
            else:
                row[pos] = old_val
    
    return best_rows, best_scope

# EVOLVE-BLOCK-END
