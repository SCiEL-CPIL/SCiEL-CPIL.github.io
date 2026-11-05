"""Difference triangle set (7,5): minimize the largest entry.
CSP approach: treat differences as a resource to partition across rows.
Backtracking with degree-minimization heuristic for future constraint violations."""
import time
import random
import math
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


def _count_future_slots(used, max_val, k):
    """Count how many distinct differences in [1, max_val] are still unused.
    This is a proxy for how much room remains for subsequent rows."""
    # Total possible differences from 1 to max_val
    total = max_val
    used_in_range = len(used & set(range(1, max_val + 1)))
    return total - used_in_range


def _fragmentation_score(used, max_val):
    """Count isolated gaps in the used difference set within [1, max_val].
    A gap is a consecutive run of unused values. More gaps = more fragmentation."""
    if not used:
        return 0
    used_sorted = sorted(d for d in used if 1 <= d <= max_val)
    if not used_sorted:
        return 0
    gaps = 0
    prev = used_sorted[0]
    for d in used_sorted[1:]:
        if d != prev + 1:
            gaps += 1
        prev = d
    # Also count gap before first and after last if they exist
    if used_sorted[0] > 1:
        gaps += 1
    if used_sorted[-1] < max_val:
        gaps += 1
    return gaps


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


def _backtrack_row_with_degree(k, used, max_val, rng, deadline, remaining_rows):
    """Build a single row using backtracking with degree-minimization heuristic.
    At each step, prefer candidates that leave the most future slots for remaining rows."""
    if deadline is not None and time.time() > deadline:
        return None
    
    row = [0]
    
    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        if low > high:
            return False
        
        # Generate candidates
        if high - low > 40:
            # Sample candidates: near lower end + random spread
            candidates = set(range(low, min(low + 15, high + 1)))
            for _ in range(25):
                candidates.add(rng.randint(low, high))
            candidates = list(candidates)
            rng.shuffle(candidates)
        else:
            candidates = list(range(low, high + 1))
            rng.shuffle(candidates)
        
        # Score each candidate by how well it preserves future slots
        scored_candidates = []
        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            # Estimate fragmentation after adding these diffs
            trial_used = used | new_diffs
            frag = _fragmentation_score(trial_used, max_val)
            slots = _count_future_slots(trial_used, max_val, k)
            # Prefer lower fragmentation and more remaining slots
            score = frag * 10 - slots
            scored_candidates.append((score, val, new_diffs))
        
        # Sort by score (lower is better) and try in that order
        scored_candidates.sort(key=lambda x: x[0])
        
        for score, val, new_diffs in scored_candidates:
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


def _construct_with_degree_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using backtracking with degree-minimization, each with elements <= max_val."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        remaining = n_rows - r - 1
        row = _backtrack_row_with_degree(k, used, max_val, rng, deadline, remaining)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows


def _simulated_annealing(rows, k, deadline, rng):
    """Simulated annealing to reduce the maximum entry."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    current_scope = _scope(rows)
    T = 20.0
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 15
        
        if low >= high:
            T *= cooling
            continue
        
        new_val = rng.randint(low, high)
        if new_val == old_val:
            T *= cooling
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
            T *= cooling
            continue
        
        new_scope = _scope([row[:pos] + [new_val] + row[pos + 1:]] + rows[:ri] + rows[ri+1:])
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    return best_rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope - 2:
                    continue
                
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                n_candidates = min(20, high - low + 1)
                candidates = rng.sample(range(low, high + 1), n_candidates)
                candidates.sort()
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
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


def _randomized_degree_backtracking(n_rows, k, deadline, rng, initial_max=120):
    """Repeatedly try to construct a valid set using randomized degree-minimization backtracking."""
    best_rows = None
    best_scope = float('inf')
    
    for max_val in range(initial_max, 104, -1):
        if time.time() > deadline:
            break
        
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            rows = _construct_with_degree_backtracking(n_rows, k, max_val, rng, deadline)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    break
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy baselines
    best_rows = None
    best_scope = float('inf')
    for seed_off in range(10):
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed_off)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Phase 2: Degree-minimization backtracking with decreasing bounds
    if time.time() < deadline - 50.0:
        bt_rows, bt_scope = _randomized_degree_backtracking(n_rows, k, deadline - 50.0, rng, initial_max=min(best_scope, 120))
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 20.0:
        sa_rows = _simulated_annealing(best_rows, k, deadline - 20.0, rng)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Local search
    if time.time() < deadline - 10.0:
        improved = _local_search_improve(best_rows, k, deadline - 10.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 5: Final refinement
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
