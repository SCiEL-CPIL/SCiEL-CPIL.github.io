import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Combine bitmask-based difference tracking with randomized
    backtracking and simulated annealing refinement.
    """
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    lower_bound = total_diffs_needed
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try to find optimal or near-optimal solution via backtracking
    # with bitmask difference tracking
    max_scope_to_try = lower_bound + 30
    
    for scope in range(lower_bound, min(max_scope_to_try, lower_bound + 15) + 1):
        if time.time() > DEADLINE - 30:
            break
        
        # Use bitmask for difference tracking - much faster than set
        # Differences are in range [1, scope], so we need scope bits
        # Python ints can handle arbitrary precision, so this is fine
        
        found = _solve_with_backtracking(scope, n_rows, k, num_elements, 
                                         time.time() + 3.0)
        if found is not None:
            # Verify
            if _is_valid(found, n_rows, k):
                s = max(r[-1] for r in found)
                if s < best_scope:
                    best_scope = s
                    best_rows = found
                break
    
    # Phase 2: If we didn't find a great solution, use greedy + SA refinement
    if best_rows is None or best_scope > lower_bound + 10:
        if time.time() < DEADLINE - 20:
            # Generate initial solution with greedy
            greedy_rows = _greedy_construct(n_rows, k, 42)
            greedy_scope = max(r[-1] for r in greedy_rows) if greedy_rows else float('inf')
            
            if greedy_scope < best_scope:
                best_scope = greedy_scope
                best_rows = greedy_rows
            
            # Refine with simulated annealing
            if best_rows is not None and time.time() < DEADLINE - 5:
                refined = _simulated_annealing(best_rows, n_rows, k, 
                                               start_time, DEADLINE)
                if refined is not None:
                    ref_scope = max(r[-1] for r in refined)
                    if ref_scope < best_scope:
                        best_scope = ref_scope
                        best_rows = refined
    
    # Phase 3: Final polish - try to reduce the maximum element
    if best_rows is not None and time.time() < DEADLINE - 2:
        polished = _polish_max(best_rows, n_rows, k, DEADLINE)
        if polished is not None:
            pol_scope = max(r[-1] for r in polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    # Fallback: simple greedy if nothing else worked
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
    
    return best_rows

def _is_valid(rows, n_rows, k):
    """Check if the construction is valid."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
    return len(rows) == n_rows

def _solve_with_backtracking(target_scope, n_rows, k, num_elements, time_limit):
    """
    Solve using randomized backtracking with bitmask difference tracking.
    Returns list of rows if found, None otherwise.
    """
    rng = random.Random(target_scope * 7919 + 13)
    used_mask = 0  # Bitmask: bit d is set if difference d is used
    
    def _can_add_diffs(diff_list):
        """Check if all diffs in diff_list are available (not in used_mask)."""
        for d in diff_list:
            if d > target_scope:
                return False
            if (used_mask >> d) & 1:
                return False
        return True
    
    def _add_diffs(diff_list):
        nonlocal used_mask
        for d in diff_list:
            used_mask |= (1 << d)
    
    def _remove_diffs(diff_list):
        nonlocal used_mask
        for d in diff_list:
            used_mask &= ~(1 << d)
    
    def build_row(pos, last_val, current_row):
        if time.time() > time_limit:
            return False
        
        if pos == num_elements:
            return True
        
        # Generate candidates
        candidates = list(range(last_val + 1, target_scope + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            # Compute new differences
            new_diffs = [val - current_row[i] for i in range(len(current_row))]
            
            if _can_add_diffs(new_diffs):
                current_row.append(val)
                _add_diffs(new_diffs)
                
                if build_row(pos + 1, val, current_row):
                    return True
                else:
                    current_row.pop()
                    _remove_diffs(new_diffs)
        
        return False
    
    rows = []
    for r in range(n_rows):
        if time.time() > time_limit:
            return None
        
        current_row = [0]
        if build_row(1, 0, current_row):
            rows.append(current_row)
        else:
            return None
    
    return rows

def _greedy_construct(n_rows, k, seed):
    """Greedy construction with packing heuristic."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            best_x = None
            best_score = float('inf')
            
            # Try a range of candidate values
            last_x = row[-1]
            for x in range(last_x + 1, last_x + 200):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                
                # Score: prefer values that pack tightly
                # Measure fragmentation
                score = 0.0
                for d in new_diffs:
                    # Distance to nearest used diff
                    min_dist = min((abs(d - u) for u in used), default=abs(d))
                    score += min_dist
                score /= len(new_diffs)
                score += 0.05 * x  # Slight preference for smaller x
                
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # Fallback: just pick the next valid x
                x = last_x + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        best_x = x
                        break
                    x += 1
            
            row.append(best_x)
            new_diffs = {best_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """Simulated annealing to refine the solution."""
    rng = random.Random(12345)
    rows = [list(r) for r in initial_rows]
    current_scope = max(r[-1] for r in rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diff sets
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    # Global diff count
    diff_count = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _row_valid(new_diffs, exclude_idx):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[exclude_idx]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit(row_idx, new_row, new_diffs):
        nonlocal rows, row_diffs, diff_count
        for d in row_diffs[row_idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[row_idx] = new_row
        row_diffs[row_idx] = new_diffs
    
    T = 10.0
    total_steps = 30000
    cool_rate = (0.1 / T) ** (1.0 / total_steps)
    
    for step in range(total_steps):
        if time.time() > deadline - 10:
            break
        
        # Choose a row to modify
        row_idx = rng.randint(0, n_rows - 1)
        
        # Try to modify the row by changing one element
        if len(rows[row_idx]) > 1:
            pos = rng.randint(1, len(rows[row_idx]) - 1)  # Don't change the 0
            old_val = rows[row_idx][pos]
            
            # Try nearby values
            for delta in [1, -1, 2, -2, 3, -3, 5, -5]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if new_val >= rows[row_idx][pos + 1] if pos + 1 < len(rows[row_idx]) else False:
                    continue
                if new_val <= rows[row_idx][pos - 1]:
                    continue
                
                # Compute new diffs
                new_row = list(rows[row_idx])
                new_row[pos] = new_val
                new_diffs = set()
                n = len(new_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                
                if _row_valid(new_diffs, row_idx):
                    new_scope = max(r[-1] for r in rows if r is not new_row) 
                    new_scope = max(new_scope, new_row[-1])
                    old_scope = max(r[-1] for r in rows)
                    
                    delta_scope = new_scope - old_scope
                    if delta_scope <= 0 or rng.random() < math.exp(-delta_scope / T):
                        _commit(row_idx, new_row, new_diffs)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                        break
        
        T *= cool_rate
    
    return best_rows if best_scope < current_scope else None

def _polish_max(rows, n_rows, k, deadline):
    """Try to reduce the maximum element."""
    rng = random.Random(999)
    rows = [list(r) for r in rows]
    best_scope = max(r[-1] for r in rows)
    best_rows = [list(r) for r in rows]
    
    # Find the row with the max element
    max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
    
    # Try to reduce the last element of that row
    for _ in range(1000):
        if time.time() > deadline - 2:
            break
        
        row = rows[max_row_idx]
        last_val = row[-1]
        
        # Try reducing last_val
        for new_last in range(last_val - 1, row[-2], -1):
            new_row = list(row)
            new_row[-1] = new_last
            
            # Check validity
            new_diffs = set()
            n = len(new_row)
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            
            # Check against other rows
            valid = True
            for other_idx in range(n_rows):
                if other_idx == max_row_idx:
                    continue
                other_diffs = set()
                m = len(rows[other_idx])
                for i in range(m):
                    for j in range(i + 1, m):
                        other_diffs.add(rows[other_idx][j] - rows[other_idx][i])
                if new_diffs & other_diffs:
                    valid = False
                    break
            
            if valid:
                new_scope = max(r[-1] for r in rows if r is not new_row)
                new_scope = max(new_scope, new_row[-1])
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    rows[max_row_idx] = new_row
                    break
    
    return best_rows if best_scope < max(r[-1] for r in rows) else None

# EVOLVE-BLOCK-END
