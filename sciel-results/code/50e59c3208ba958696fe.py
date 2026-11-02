<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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
    return rows

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Return the set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if a row is a Sidon set (all pairwise differences distinct)."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return len(diffs) == len(set(diffs))

def _check_row_valid(row, k):
    """Check if a single row is strictly increasing and starts at 0."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def _backtrack_row(used_diffs, k, M, rng=None):
    """
    Try to build a single row of k+1 increasing integers starting at 0,
    all <= M, with no difference colliding with used_diffs and no internal
    duplicate differences (Sidon).
    Returns the row as a list, or None if impossible.
    """
    if rng is None:
        rng = random
    
    row = [0]
    used_local = set()
    
    def dfs(idx, last_val):
        if idx == k:
            return True
        # Try next values from last_val+1 to M
        start = last_val + 1
        end = M + 1
        # Randomize order for diversity
        candidates = list(range(start, end))
        rng.shuffle(candidates)
        for v in candidates:
            # Check differences with all previous elements
            new_diffs = set()
            conflict = False
            for i in range(idx):
                d = v - row[i]
                if d in used_diffs or d in used_local:
                    conflict = True
                    break
                new_diffs.add(d)
            if conflict:
                continue
            # Check internal uniqueness (Sidon within this row)
            # new_diffs already checked against used_local, but we need
            # to ensure no two pairs in the full row produce same diff
            # Since we check against used_local (all diffs so far), this is fine
            row.append(v)
            used_local.update(new_diffs)
            if dfs(idx + 1, v):
                return True
            row.pop()
            for d in new_diffs:
                used_local.discard(d)
        return False
    
    if dfs(1, 0):
        return row[:]
    return None

def _resolve_row_given_others(others_diffs, k, M, rng=None):
    """
    Given the set of differences used by other rows, try to construct a valid
    row of k+1 elements starting at 0, max <= M, avoiding all those differences.
    """
    if rng is None:
        rng = random
    return _backtrack_row(others_diffs, k, M, rng)

def _try_construct_at_M(n_rows, k, M, rng=None):
    """
    Try to construct n_rows valid rows, each with k+1 elements starting at 0,
    all <= M, with globally distinct differences.
    Returns list of rows or None.
    """
    if rng is None:
        rng = random
    
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        row = _resolve_row_given_others(used_diffs, k, M, rng)
        if row is None:
            return None
        rows.append(row)
        used_diffs |= _row_diffs_set(row)
    
    return rows

def _ils_search(n_rows, k, time_limit=90.0, seed=42):
    """
    Iterated Local Search:
    - Start from a greedy solution
    - Perturb by re-solving one or more rows with randomization
    - Local search: try to reduce max entry by re-solving rows at smaller M
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(seed)
    
    # Initial greedy solution
    best = _build_greedy(n_rows, k)
    best_val = _max_entry(best)
    
    # Try to improve by re-solving rows
    cur = [r[:] for r in best]
    cur_val = best_val
    
    # Local search: try to reduce max by re-solving the row(s) with max entry
    def local_search(rows, val, rng):
        rows = [r[:] for r in rows]
        val = _max_entry(rows)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find rows with maximum entry
            max_rows = [i for i, r in enumerate(rows) if max(r) == val]
            # Try re-solving one of them at a slightly smaller M
            for ri in max_rows:
                if time.time() >= deadline:
                    break
                # Try M from val-1 down to val-5
                for trial_M in range(val - 1, max(val - 6, 10), -1):
                    others_diffs = set()
                    for j, r in enumerate(rows):
                        if j != ri:
                            others_diffs |= _row_diffs_set(r)
                    new_row = _resolve_row_given_others(others_diffs, k, trial_M, rng)
                    if new_row is not None:
                        rows[ri] = new_row
                        new_val = _max_entry(rows)
                        if new_val < val:
                            val = new_val
                            improved = True
                            break
                    if improved:
                        break
        return rows, val
    
    # ILS loop
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Perturb: re-solve a random row with randomization
        ri = rng.randrange(n_rows)
        others_diffs = set()
        for j, r in enumerate(cur):
            if j != ri:
                others_diffs |= _row_diffs_set(r)
        
        # Try to re-solve at current max or slightly larger
        trial_M = cur_val + rng.randint(0, 5)
        new_row = _resolve_row_given_others(others_diffs, k, trial_M, rng)
        
        if new_row is not None:
            cur[ri] = new_row
            cur_val = _max_entry(cur)
        
        # Local search
        cur, cur_val = local_search(cur, cur_val, rng)
        
        if cur_val < best_val:
            best_val = cur_val
            best = [r[:] for r in cur]
            no_improve = 0
        else:
            no_improve += 1
    
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    random.seed(42)
    
    # ILS with row-resolution representation
    best = _ils_search(n_rows, k, time_limit=100.0, seed=42)
    best_val = _max_entry(best)
    
    # Ensure validity
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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
    return rows

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Return the set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _all_diffs_set(rows):
    """Return the set of all differences across all rows."""
    s = set()
    for row in rows:
        s |= _row_diffs_set(row)
    return s

def _backtrack_row(used_diffs, k, M, rng=None, max_nodes=50000):
    """
    Try to build a single row of k+1 increasing integers starting at 0,
    all <= M, with no difference colliding with used_diffs and no internal
    duplicate differences (Sidon).
    Returns the row as a list, or None if impossible.
    """
    if rng is None:
        rng = random
    
    row = [0]
    used_local = set()
    nodes_visited = [0]
    
    def dfs(idx, last_val):
        if idx == k:
            return True
        if nodes_visited[0] >= max_nodes:
            return False
        # Try next values from last_val+1 to M
        start = last_val + 1
        end = M + 1
        # Randomize order for diversity
        candidates = list(range(start, end))
        rng.shuffle(candidates)
        for v in candidates:
            nodes_visited[0] += 1
            if nodes_visited[0] >= max_nodes:
                return False
            # Check differences with all previous elements
            new_diffs = set()
            conflict = False
            for i in range(idx):
                d = v - row[i]
                if d in used_diffs or d in used_local:
                    conflict = True
                    break
                new_diffs.add(d)
            if conflict:
                continue
            row.append(v)
            used_local.update(new_diffs)
            if dfs(idx + 1, v):
                return True
            row.pop()
            for d in new_diffs:
                used_local.discard(d)
        return False
    
    if dfs(1, 0):
        return row[:]
    return None

def _resolve_row_given_others(others_diffs, k, M, rng=None, max_nodes=50000):
    """
    Given the set of differences used by other rows, try to construct a valid
    row of k+1 elements starting at 0, max <= M, avoiding all those differences.
    """
    if rng is None:
        rng = random
    return _backtrack_row(others_diffs, k, M, rng, max_nodes)

def _sa_move_resolve_row(cur_rows, ri, cur_val, rng, deadline):
    """
    Move type 1: Re-solve row ri given all other rows' differences.
    Returns (new_rows, new_val) or None if move not possible.
    """
    others_diffs = set()
    for j, r in enumerate(cur_rows):
        if j != ri:
            others_diffs |= _row_diffs_set(r)
    
    # Try to re-solve at current max or slightly larger
    trial_M = cur_val + rng.randint(0, 3)
    new_row = _resolve_row_given_others(others_diffs, k, trial_M, rng, max_nodes=30000)
    
    if new_row is not None:
        new_rows = [r[:] for r in cur_rows]
        new_rows[ri] = new_row
        new_val = _max_entry(new_rows)
        return new_rows, new_val
    return None

def _sa_move_swap_rows(cur_rows, ri, rj, cur_val, rng, deadline):
    """
    Move type 2: Swap two rows to potentially allow better packing.
    This is a neutral move that changes the order of resolution.
    """
    if ri == rj:
        return None
    new_rows = [r[:] for r in cur_rows]
    new_rows[ri], new_rows[rj] = new_rows[rj], new_rows[ri]
    new_val = _max_entry(new_rows)
    return new_rows, new_val

def _sa_move_partial_repair(cur_rows, ri, cur_val, rng, deadline):
    """
    Move type 3: Partially repair row ri by trying to reduce its max entry
    while keeping the first few elements fixed.
    """
    row = cur_rows[ri]
    # Try to rebuild the row with a smaller max
    target_max = max(row) - rng.randint(1, 3)
    if target_max < len(row):
        target_max = len(row)
    
    # Get differences from other rows
    others_diffs = set()
    for j, r in enumerate(cur_rows):
        if j != ri:
            others_diffs |= _row_diffs_set(r)
    
    # Try to rebuild with constraint that last element <= target_max
    # We'll try to find a new row that fits in [0, target_max]
    new_row = _resolve_row_given_others(others_diffs, k, target_max, rng, max_nodes=20000)
    
    if new_row is not None:
        new_rows = [r[:] for r in cur_rows]
        new_rows[ri] = new_row
        new_val = _max_entry(new_rows)
        return new_rows, new_val
    return None

def _sa_run(n_rows, k, time_limit, seed, initial_rows=None):
    """
    Simulated annealing search with multiple move types.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(seed)
    
    # Initialize
    if initial_rows is None:
        cur_rows = _build_greedy(n_rows, k)
    else:
        cur_rows = [r[:] for r in initial_rows]
    cur_val = _max_entry(cur_rows)
    best_rows = [r[:] for r in cur_rows]
    best_val = cur_val
    
    # SA parameters
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    steps_per_temp = 20
    
    current_T = T
    
    while time.time() < deadline and current_T > T_min:
        for _ in range(steps_per_temp):
            if time.time() >= deadline:
                break
            
            # Choose move type
            move_type = rng.random()
            ri = rng.randrange(n_rows)
            
            if move_type < 0.5:
                # Move type 1: Resolve row
                result = _sa_move_resolve_row(cur_rows, ri, cur_val, rng, deadline)
            elif move_type < 0.7:
                # Move type 2: Swap rows
                rj = rng.randrange(n_rows)
                result = _sa_move_swap_rows(cur_rows, ri, rj, cur_val, rng, deadline)
            else:
                # Move type 3: Partial repair
                result = _sa_move_partial_repair(cur_rows, ri, cur_val, rng, deadline)
            
            if result is None:
                continue
            
            new_rows, new_val = result
            
            # Acceptance criterion
            if new_val < cur_val:
                # Always accept improvement
                cur_rows = new_rows
                cur_val = new_val
                if cur_val < best_val:
                    best_val = cur_val
                    best_rows = [r[:] for r in cur_rows]
            else:
                # Accept worse with probability exp(-delta/T)
                delta = new_val - cur_val
                if delta == 0:
                    # Neutral move, accept with some probability
                    if rng.random() < 0.3:
                        cur_rows = new_rows
                        cur_val = new_val
                else:
                    prob = math.exp(-delta / current_T)
                    if rng.random() < prob:
                        cur_rows = new_rows
                        cur_val = new_val
        
        # Cool down
        current_T *= cooling_rate
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    random.seed(42)
    
    # Run multiple SA restarts with different seeds
    best_overall = None
    best_val_overall = float('inf')
    
    # First, get a greedy solution as baseline
    greedy = _build_greedy(n_rows, k)
    greedy_val = _max_entry(greedy)
    best_overall = greedy
    best_val_overall = greedy_val
    
    # Run SA with multiple seeds and time allocations
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5:
            break
        
        seed = 42 + restart * 1000
        remaining_time = deadline - time.time() - 2  # 2s safety margin
        
        if remaining_time < 10:
            break
        
        # Run SA
        sa_result = _sa_run(n_rows, k, remaining_time, seed)
        sa_val = _max_entry(sa_result)
        
        if sa_val < best_val_overall:
            best_val_overall = sa_val
            best_overall = sa_result
    
    # Ensure validity
    if not _valid(best_overall, n_rows, k):
        best_overall = _build_greedy(n_rows, k)
    
    return best_overall
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
