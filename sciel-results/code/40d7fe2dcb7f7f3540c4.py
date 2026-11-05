"""Difference triangle set (7,5): minimize the largest entry.
Structural redesign: model as difference-set assignment with constraint propagation
and randomized backtracking, plus local search on the difference-level."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _fragmentation_score(used, new_diffs):
    """Count how many new isolated gaps (fragmentation) are created by adding new_diffs."""
    if not new_diffs:
        return 0
    sorted_new = sorted(new_diffs)
    count = 0
    for d in sorted_new:
        # Check if d creates a new isolated gap: neither d-1 nor d+1 in used
        if (d - 1) not in used and (d + 1) not in used:
            count += 1
        # Also check if d-1 or d+1 is in new_diffs (they'll be added together)
        elif (d - 1) in new_diffs or (d + 1) in new_diffs:
            pass  # Not isolated
        else:
            count += 1
    return count


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with fragmentation-aware heuristic."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            best_val = None
            best_score = float('inf')
            
            # Search for good candidates
            max_search = 200
            x_start = row[-1] + 1
            x_end = x_start + max_search
            
            for x in range(x_start, x_end + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                
                # Score: prefer smaller x, less fragmentation
                frag = _fragmentation_score(used, new_diffs)
                score = x + 5 * frag
                if score < best_score:
                    best_score = score
                    best_val = x
            
            if best_val is None:
                # Fallback: find any valid x
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        best_val = x
                        break
                    x += 1
            
            row.append(best_val)
            new_diffs = {best_val - a for a in row[:-1]}
            used |= new_diffs
        rows.append(row)
    
    return rows


def _backtrack_row(k, used, max_val, rng, time_limit=None, deadline=None):
    """Try to build a single row of k+1 elements starting at 0, with all differences
    not in 'used', and all elements <= max_val. Returns row or None."""
    if deadline is not None and time.time() > deadline:
        return None
    
    row = [0]
    
    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        # Determine range of valid next values
        low = row[-1] + 1
        high = min(max_val, row[-1] + (max_val - row[-1]))  # just max_val
        
        # Shuffle candidate order for randomization
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            row.append(val)
            used.update(new_diffs)
            if backtrack(idx + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_with_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using backtracking, each with elements <= max_val."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = _backtrack_row(k, used, max_val, rng, deadline=deadline)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search with simulated annealing: try to reduce the maximum entry."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    best_rows = [list(r) for r in rows]
    
    # Simulated annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.98
    iterations = 0
    
    while time.time() < deadline and T > T_min:
        iterations += 1
        current_scope = _scope(rows)
        
        # Pick a random row and position to modify
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 10
        
        if low >= high:
            T *= cooling
            continue
        
        # Generate candidates: prefer values that reduce scope
        candidates = set()
        # Add values near old_val
        for delta in range(-15, 16):
            v = old_val + delta
            if low <= v <= high:
                candidates.add(v)
        # Add values that would reduce the max if this is the max element
        if old_val >= current_scope:
            for dv in range(1, 20):
                v = current_scope - dv
                if low <= v <= high:
                    candidates.add(v)
        
        if not candidates:
            T *= cooling
            continue
        
        # Evaluate all candidates
        best_new_val = None
        best_new_scope = current_scope
        best_new_diffs = None
        best_old_diffs = None
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            
            # Compute old diffs involving pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            # Compute new diffs involving pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            # Check for conflicts
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            # Calculate new scope
            temp_row = row[:pos] + [new_val] + row[pos + 1:]
            new_scope = _scope([r if r is not row else temp_row for r in rows])
            
            if new_scope < best_new_scope:
                best_new_scope = new_scope
                best_new_val = new_val
                best_new_diffs = new_diffs
                best_old_diffs = old_diffs
        
        if best_new_val is not None:
            delta = best_new_scope - current_scope
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 1e-9))):
                row[pos] = best_new_val
                used -= best_old_diffs
                used |= best_new_diffs
                
                if best_new_scope < best_scope:
                    best_scope = best_new_scope
                    best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    return best_rows


def _difference_swap_mutation(rows, k, used, num_swaps=None, rng=None):
    """Perform difference-swap moves to reduce the max entry."""
    if rng is None:
        rng = random.Random()
    if num_swaps is None:
        num_swaps = rng.randint(1, 3)
    
    rows = [list(r) for r in rows]
    used = set(used)
    n_rows = len(rows)
    
    for _ in range(num_swaps):
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 20
        
        if low >= high:
            continue
        
        best_new_val = None
        best_score = float('inf')
        
        candidates = set()
        for _ in range(30):
            candidates.add(rng.randint(low, high))
        current_max = max(row)
        if pos == len(row) - 1:
            for dv in range(1, 8):
                candidates.add(max(low, current_max - dv))
        
        for new_val in candidates:
            if new_val == old_val or new_val < low or new_val > high:
                continue
            
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
            old_row_max = max(row)
            score = new_row_max - old_row_max
            if score < best_score:
                best_score = score
                best_new_val = new_val
        
        if best_new_val is not None:
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(best_new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - best_new_val)
            
            row[pos] = best_new_val
            used -= old_diffs
            used |= new_diffs
    
    return rows, used


def _randomized_backtracking(n_rows, k, deadline, rng, initial_max=120):
    """Repeatedly try to construct a valid set using randomized backtracking,
    with decreasing max_val bounds."""
    best_rows = None
    best_scope = float('inf')
    
    # Try various max_val bounds
    for max_val in range(initial_max, 104, -1):
        if time.time() > deadline:
            break
        
        # Multiple attempts at this max_val
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            rows = _construct_with_backtracking(n_rows, k, max_val, rng, deadline)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    break  # Found a valid set at this max_val, try lower
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy baseline with fragmentation heuristic
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Randomized backtracking with decreasing bounds
    if time.time() < deadline - 40.0:
        bt_rows, bt_scope = _randomized_backtracking(n_rows, k, deadline - 40.0, rng, initial_max=best_scope)
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Local search with simulated annealing to improve
    if time.time() < deadline - 10.0:
        improved = _local_search_improve(best_rows, k, deadline - 10.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Final refinement with difference-swap mutations
    if time.time() < deadline - 3.0:
        used = _all_diffs(best_rows)
        for _ in range(20):
            if time.time() > deadline - 3.0:
                break
            refined, used = _difference_swap_mutation(best_rows, k, used, num_swaps=rng.randint(2, 5), rng=rng)
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
