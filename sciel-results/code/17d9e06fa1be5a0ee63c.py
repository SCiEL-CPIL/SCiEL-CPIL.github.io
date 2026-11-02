"""Constraint-satisfaction search for the difference-triangle problem."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
    used, rows = set(), []
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


def _scope(rows):
    return max(max(r) for r in rows)


def _compute_row_diffs(row):
    """Compute set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_valid(rows):
    """Check if all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True


def _randomized_greedy(n_rows, k, rng, bias=0.5, window=3):
    """Randomized greedy construction with configurable bias and window."""
    used, rows = set(), []
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            # Look at a window of candidates
            candidates = []
            for cand in range(x + 1, x + 1 + window):
                new = {cand - a for a in row}
                if not new & used:
                    # Score: sum of new differences (prefer small)
                    candidates.append((sum(new), cand))
            if candidates:
                candidates.sort()
                # Pick with some randomness
                if rng.random() < bias:
                    chosen = candidates[0][1]
                else:
                    chosen = candidates[rng.randrange(len(candidates))][1]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
                found = True
            else:
                # No candidate in window; extend
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                    found = True
            if not found:
                # Shouldn't happen since we always advance x
                pass
        rows.append(row)
    return rows


def _local_search(rows, k, deadline, rng):
    """
    Local search: try to reduce scope by shifting entries down.
    For each row, for each position, try to decrease the entry while maintaining
    strict increase and no difference collisions.
    """
    best_rows = [list(r) for r in rows]
    best_sc = _scope(best_rows)

    def compute_all_diffs(rws):
        """Compute the set of all differences across all rows."""
        all_diffs = set()
        for r in rws:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.add(r[j] - r[i])
        return all_diffs

    improved = True
    t0 = time.time()
    while improved and time.time() < deadline:
        improved = False
        # Compute all diffs once for efficiency
        all_diffs = compute_all_diffs(best_rows)
        
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            # Try to reduce each entry (from last to first to preserve increase)
            for pos in range(len(row) - 1, 0, -1):
                lower = row[pos - 1] + 1
                upper = row[pos] - 1
                if upper < lower:
                    continue
                # Try candidate values from lower upward, pick the smallest valid
                for cand in range(lower, upper + 1):
                    if cand == row[pos]:
                        break
                    # Check: does setting row[pos]=cand cause any duplicate difference?
                    old = row[pos]
                    row[pos] = cand
                    
                    # Recompute diffs for this row
                    row_diffs = _compute_row_diffs(row)
                    
                    # Check for conflicts within this row
                    valid = True
                    if len(row_diffs) < len(row) * (len(row) - 1) // 2:
                        valid = False
                    else:
                        # Check against other rows
                        for d in row_diffs:
                            # Remove this row's diffs from all_diffs to check
                            pass
                    
                    # More efficient: check each diff in row_diffs
                    # First, remove this row's original diffs from all_diffs
                    orig_diffs = _compute_row_diffs([0] + list(row[:pos]) + [old] + list(row[pos+1:]))
                    
                    # Check new diffs
                    for d in row_diffs:
                        if d in all_diffs and d not in orig_diffs:
                            valid = False
                            break
                    
                    if valid:
                        new_sc = _scope(best_rows)
                        if new_sc <= best_sc:
                            # Update all_diffs
                            all_diffs = all_diffs - orig_diffs
                            all_diffs |= row_diffs
                            best_sc = new_sc
                            improved = True
                        else:
                            row[pos] = old
                    else:
                        row[pos] = old
                        break  # no point trying larger cand in this direction
    
    return best_rows, best_sc


def _simulated_annealing(rows, k, deadline, rng):
    """
    Simulated annealing: perturb a row by changing one entry, accept if valid
    and improves or with probability exp(-delta/T).
    """
    best_rows = [list(r) for r in rows]
    best_sc = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_sc = _scope(current_rows)
    
    def compute_all_diffs(rws):
        all_diffs = set()
        for r in rws:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.add(r[j] - r[i])
        return all_diffs
    
    current_diffs = compute_all_diffs(current_rows)
    T = max(1, current_sc)  # Initial temperature
    T_min = 1.0
    cooling = 0.995
    
    t0 = time.time()
    steps = 0
    
    while time.time() < deadline and T > T_min:
        steps += 1
        
        # Pick a random row and position to perturb
        ri = rng.randrange(len(current_rows))
        pos = rng.randrange(1, len(current_rows[ri]))
        row = current_rows[ri]
        
        # Pick a new value: either shift by +/- 1 to +/- 3
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        new_val = row[pos] + delta
        
        # Check bounds
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue
        if new_val < 0:
            continue
        
        old_val = row[pos]
        row[pos] = new_val
        
        # Check validity
        new_row_diffs = _compute_row_diffs(row)
        
        # Check for internal conflicts
        valid = len(new_row_diffs) == len(row) * (len(row) - 1) // 2
        
        if valid:
            # Check against other rows
            orig_row_diffs = _compute_row_diffs([0] + list(row[:pos]) + [old_val] + list(row[pos+1:]))
            for d in new_row_diffs:
                if d in current_diffs and d not in orig_row_diffs:
                    valid = False
                    break
        
        if valid:
            # Update diffs
            orig_row_diffs = _compute_row_diffs([0] + list(row[:pos]) + [old_val] + list(row[pos+1:]))
            current_diffs = current_diffs - orig_row_diffs
            current_diffs |= new_row_diffs
            
            new_sc = _scope(current_rows)
            delta_sc = new_sc - current_sc
            
            if delta_sc <= 0:
                # Accept: improvement
                current_sc = new_sc
                if new_sc < best_sc:
                    best_sc = new_sc
                    best_rows = [list(r) for r in current_rows]
            else:
                # Maybe accept: worse
                prob = math.exp(-delta_sc / T)
                if rng.random() < prob:
                    current_sc = new_sc
                else:
                    # Reject: revert
                    row[pos] = old_val
                    current_diffs = current_diffs - new_row_diffs
                    current_diffs |= orig_row_diffs
        else:
            # Revert
            row[pos] = old_val
        
        T *= cooling
    
    return best_rows, best_sc


def construct(n_rows, k):
    deadline = time.time() + 100.0  # safety margin before 150s kill
    rng = random.Random(12345)
    
    # Phase 1: Try multiple randomized greedy constructions
    best_rows = _greedy_fallback(n_rows, k)
    best_sc = _scope(best_rows)
    
    t1 = time.time()
    phase1_deadline = deadline - 40.0
    
    # Try different bias and window settings
    for bias in [0.3, 0.5, 0.7, 0.9]:
        for window in [2, 3, 5]:
            if time.time() > phase1_deadline:
                break
            rows = _randomized_greedy(n_rows, k, rng, bias, window)
            sc = _scope(rows)
            if sc < best_sc:
                best_sc = sc
                best_rows = rows
    
    # Phase 2: Local search to polish
    if time.time() < deadline - 20.0:
        phase2_deadline = deadline - 15.0
        best_rows, best_sc = _local_search(best_rows, k, phase2_deadline, rng)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        phase3_deadline = deadline - 2.0
        best_rows, best_sc = _simulated_annealing(best_rows, k, phase3_deadline, rng)
    
    return best_rows

# EVOLVE-BLOCK-END
