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


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(50):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
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
    """Local search: try to reduce the maximum entry by modifying individual elements.
    Works at the difference level: for each element position, try alternative values."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    # Build a map from difference to (row_idx, i, j) for quick lookup
    diff_map = {}
    for ri, row in enumerate(rows):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_map[d] = (ri, i, j)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        # Try to reduce each element that is at the current maximum
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):  # from last to first
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope:
                    continue
                
                # Try reducing this value
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                # Try values from high down to low (prefer smaller max)
                candidates = list(range(high, low - 1, -1))
                rng.shuffle(candidates)
                # But prefer values close to high first (smaller change)
                candidates.sort(key=lambda v: abs(v - old_val))
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
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
                    
                    # Accept this change
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
    
    return rows


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
    
    # Phase 1: Quick greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Randomized backtracking with decreasing bounds
    if time.time() < deadline - 30.0:
        bt_rows, bt_scope = _randomized_backtracking(n_rows, k, deadline - 30.0, rng, initial_max=best_scope)
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Final refinement with difference-swap mutations
    if time.time() < deadline - 2.0:
        used = _all_diffs(best_rows)
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            refined, used = _difference_swap_mutation(best_rows, k, used, num_swaps=rng.randint(2, 5), rng=rng)
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
