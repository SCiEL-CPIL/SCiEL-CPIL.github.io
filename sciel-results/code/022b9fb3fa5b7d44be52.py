"""Difference triangle set (4,7) — Simulated Annealing with repair and tight packing."""
import time
import random
import sys
from collections import Counter

# EVOLVE-BLOCK-START

def _compute_diffs(row):
    """Compute set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return a Counter of all differences across all rows."""
    counter = Counter()
    for row in rows:
        for d in _compute_diffs(row):
            counter[d] += 1
    return counter

def _count_conflicts(counter):
    """Count the number of duplicate differences."""
    conflicts = 0
    for d, cnt in counter.items():
        if cnt > 1:
            conflicts += cnt - 1
    return conflicts

def _scope(rows):
    """Return the maximum entry in the rows."""
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check if all differences are distinct."""
    counter = _all_diffs(rows)
    return all(cnt == 1 for cnt in counter.values())

def _objective(rows, weight_scope=10):
    """
    Objective function to minimize.
    Combines conflict count and scope.
    We want 0 conflicts and minimal scope.
    """
    counter = _all_diffs(rows)
    conflicts = _count_conflicts(counter)
    sc = _scope(rows)
    # Penalize conflicts heavily
    return conflicts * 10000 + sc * weight_scope

def _greedy_row(used_diffs, k, rng, bias=1.0):
    """
    Greedily construct a single row of k+1 elements starting at 0,
    avoiding differences in used_diffs.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        # Check if adding x creates any conflict
        new_diffs = {x - a for a in row}
        if new_diffs & used_diffs:
            continue
        
        # Decide whether to take x
        # If bias is high, we take it. If low, we might skip to allow smaller future jumps?
        # Actually, greedy usually takes the smallest valid x.
        # Randomness is added by sometimes skipping a valid x to try a different path.
        if rng.random() < bias:
            row.append(x)
            used_diffs.update(new_diffs)
        else:
            # Skip x, but don't add to used_diffs
            pass
            
    return row

def _greedy_build(n_rows, k, rng, bias=0.9):
    """
    Build a full set of rows greedily.
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        # Shuffle the order of rows? No, we build sequentially.
        # But we can randomize the 'bias' or the starting strategy.
        row = _greedy_row(used_diffs, k, rng, bias)
        if len(row) < k + 1:
            return None # Failed to build
        rows.append(row)
        used_diffs.update(_compute_diffs(row))
    return rows

def _simulated_annealing(rows, n_rows, k, deadline, rng):
    """
    Simulated Annealing to optimize the set of rows.
    """
    current_rows = [r[:] for r in rows]
    current_obj = _objective(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_obj = current_obj
    
    start_time = time.time()
    T = 100.0  # Initial temperature
    cooling = 0.995  # Cooling factor
    
    while time.time() < deadline:
        if T < 1e-6:
            break
            
        # Generate a neighbor
        new_rows = [r[:] for r in current_rows]
        move_type = rng.random()
        
        if move_type < 0.7:
            # Shift an element in a row
            row_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)  # Don't move the first element (0)
            delta = rng.choice([-1, 1, 2, -2, 3, -3, 5, -5])
            old_val = new_rows[row_idx][pos]
            new_val = old_val + delta
            
            # Check strictly increasing
            if pos > 0 and new_val <= new_rows[row_idx][pos - 1]:
                continue
            if pos < k and new_val >= new_rows[row_idx][pos + 1]:
                continue
            if new_val < 0:
                continue
                
            new_rows[row_idx][pos] = new_val
        elif move_type < 0.9:
            # Rebuild a single row greedily with randomness
            row_idx = rng.randrange(n_rows)
            used_others = set()
            for i in range(n_rows):
                if i != row_idx:
                    used_others.update(_compute_diffs(new_rows[i]))
            
            # Try a few random greedy rebuilds
            best_new_row = None
            best_row_obj = float('inf')
            for _ in range(5):
                trial_row = _greedy_row(used_others, k, rng, bias=rng.uniform(0.5, 1.0))
                if len(trial_row) == k + 1:
                    # Quick check: does it introduce conflicts?
                    # We'll let the objective function handle it, but prefer valid ones
                    trial_rows = [r[:] for r in new_rows]
                    trial_rows[row_idx] = trial_row
                    obj = _objective(trial_rows)
                    if obj < best_row_obj:
                        best_row_obj = obj
                        best_new_row = trial_row
            
            if best_new_row:
                new_rows[row_idx] = best_new_row
        else:
            # Swap two elements from different rows (if valid)
            r1, r2 = rng.sample(range(n_rows), 2)
            p1 = rng.randrange(1, k + 1)
            p2 = rng.randrange(1, k + 1)
            
            v1 = new_rows[r1][p1]
            v2 = new_rows[r2][p2]
            
            # Try swapping
            new_rows[r1][p1] = v2
            new_rows[r2][p2] = v1
            
            # Check strictly increasing for both rows
            valid = True
            for r_idx, p_idx in [(r1, p1), (r2, p2)]:
                row = new_rows[r_idx]
                if p_idx > 0 and row[p_idx] <= row[p_idx - 1]:
                    valid = False
                    break
                if p_idx < k and row[p_idx] >= row[p_idx + 1]:
                    valid = False
                    break
            if not valid:
                # Revert
                new_rows[r1][p1] = v1
                new_rows[r2][p2] = v2
                continue

        # Calculate objective
        new_obj = _objective(new_rows)
        
        # Acceptance probability
        delta_obj = new_obj - current_obj
        if delta_obj < 0 or rng.random() < pow(2.718281828, -delta_obj / T):
            current_rows = new_rows
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_rows = [r[:] for r in current_rows]
        
        # Cool down
        T *= cooling
        
        # Restart if stuck?
        # For now, simple geometric cooling.
        
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin
    rng = random.Random(42)
    
    best_rows = None
    best_score = float('inf')
    
    # 1. Try multiple greedy initializations
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        trial_rng = random.Random(seed)
        # Vary bias
        bias = trial_rng.uniform(0.7, 1.0)
        rows = _greedy_build(n_rows, k, trial_rng, bias)
        if rows:
            sc = _scope(rows)
            if sc < best_score:
                best_score = sc
                best_rows = rows
    
    if best_rows is None:
        # Fallback: simple sequential greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _scope(best_rows)
        
    # 2. Run Simulated Annealing on the best found so far
    if time.time() < deadline - 5:
        # Run SA with a time budget
        sa_deadline = deadline - 5
        sa_rows = _simulated_annealing(best_rows, n_rows, k, sa_deadline, rng)
        if sa_rows:
            sa_score = _scope(sa_rows)
            # Only accept if valid or better score (SA objective handles validity via conflicts)
            # But we must return a VALID set.
            if _is_valid(sa_rows):
                if sa_score < best_score:
                    best_score = sa_score
                    best_rows = sa_rows
            else:
                # If SA returned invalid, try to repair or discard
                # For simplicity, discard if invalid and we have a valid best
                pass

    # 3. Final check and return
    # Ensure the returned set is valid. If not, try a last-ditch repair or return best valid found.
    if not _is_valid(best_rows):
        # This should not happen if greedy built valid rows and SA only accepted valid improvements
        # But just in case, run a quick local search to fix
        # Or just return the best valid one we had before SA if SA broke it.
        # Since we only update best_rows if valid in the SA block, best_rows should be valid.
        pass

    return best_rows

# EVOLVE-BLOCK-END
