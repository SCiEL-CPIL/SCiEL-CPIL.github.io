import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 148
    
    # 1. Baseline: greedy construction
    rows = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in rows)
    best_rows = rows
    
    # 2. Try to improve scope using Hungarian matching approach
    # Start from current best and try to reduce
    target_scope = best_scope - 1
    
    while target_scope >= k * (k + 1) // 2 and time.time() < deadline - 5:
        # Try multiple random restarts of the Hungarian-based construction
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            solution = hungarian_based_construct(n_rows, k, target_scope, start_time, deadline)
            if solution is not None:
                # Validate
                if _validate(solution, n_rows, k):
                    best_rows = solution
                    best_scope = target_scope
                    found = True
                    break
        
        if found:
            target_scope -= 1
        else:
            # Try one more level down in case we got unlucky
            target_scope -= 1
            if time.time() > deadline - 10:
                break
    
    return best_rows

def _validate(rows, n_rows, k):
    """Check that all positive differences are distinct across all rows."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
    return True

def greedy_construct(n_rows, k):
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

def hungarian_based_construct(n_rows, k, max_scope, start_time, deadline):
    """
    Construct rows using a Hungarian matching approach.
    
    For each row, we need to choose k positions (values) from 1..max_scope.
    The constraint is that all pairwise differences must be globally unique.
    
    Strategy: Process rows one at a time. For each row, build a cost matrix
    where assigning position j to value v has cost based on:
    - Whether v creates conflicts with already-used differences
    - The maximum value used (exponential penalty)
    
    We use a simplified approach: for each row, we greedily select values
    using a priority that accounts for future flexibility, then verify.
    If verification fails, we retry with different randomization.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline - 3:
            return None
        
        # Build candidate values for this row
        # We need k+1 values: 0, v1, v2, ..., vk where 0 < v1 < v2 < ... < vk <= max_scope
        
        # Use a scoring function to select values
        # Score each potential value based on:
        # 1. Number of new differences it creates that conflict with used_diffs
        # 2. How "spread out" it is (to leave room for future rows)
        
        row = [0]
        current_max = 0
        
        for pos in range(k):
            if time.time() > deadline - 3:
                return None
            
            # Find the best next value
            best_val = None
            best_score = float('inf')
            
            # Candidate range: from current_max+1 to max_scope
            # We want to try values that minimize conflicts
            search_start = current_max + 1
            search_end = max_scope
            
            # For efficiency, we sample candidates rather than checking all
            if search_end - search_start > 50:
                # Sample candidates
                candidates = set()
                # Always include some spread-out values
                for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    candidates.add(search_start + int(frac * (search_end - search_start)))
                # Add random candidates
                for _ in range(30):
                    candidates.add(random.randint(search_start, search_end))
                # Add the minimum possible
                candidates.add(search_start)
                candidates = sorted(candidates)
            else:
                candidates = list(range(search_start, search_end + 1))
            
            for v in candidates:
                # Check if this value creates any conflicts
                conflicts = 0
                new_diffs = []
                for a in row:
                    d = v - a
                    if d in used_diffs:
                        conflicts += 1
                        break
                    new_diffs.append(d)
                
                if conflicts > 0:
                    continue
                
                # Score: prefer values that create "unique" differences
                # and leave room for future selections
                # The key insight: we want differences that are hard to reuse
                score = 0.0
                
                # Penalize if any new difference is small (easier to conflict later)
                for d in new_diffs:
                    score += 1.0 / max(d, 1)
                
                # Penalize if v is too close to max_scope (limits future options)
                remaining_positions = k - pos - 1
                if remaining_positions > 0:
                    room = max_scope - v
                    needed = remaining_positions  # at minimum, need room for remaining
                    if room < needed:
                        score += 100.0 * (needed - room)
                
                # Small random perturbation for diversity
                score += random.random() * 0.1
                
                if score < best_score:
                    best_score = score
                    best_val = v
                    best_new_diffs = new_diffs
            
            if best_val is None:
                return None
            
            row.append(best_val)
            current_max = best_val
            used_diffs.update(best_new_diffs)
        
        # Check that the row is valid and within scope
        if max(row) > max_scope:
            return None
        
        rows.append(row)
    
    # Final validation
    if _validate(rows, n_rows, k):
        return rows
    return None

def backtracking_search(n_rows, k, max_scope, deadline):
    used_mask = 0
    rows = []
    node_count = 0
    max_nodes = 100000 # Limit per search
    
    def backtrack(row_idx, current_row):
        nonlocal used_mask, node_count
        if time.time() > deadline or node_count > max_nodes:
            return False
        node_count += 1
        
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row[:])
            if backtrack(row_idx + 1, [0]):
                return True
            rows.pop()
            return False
        else:
            last = current_row[-1]
            candidates = []
            for x in range(last + 1, max_scope + 1):
                new_mask = 0
                conflict = False
                for a in current_row:
                    diff = x - a
                    bit = 1 << diff
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if conflict:
                    continue
                # Space check
                if max_scope - x < (k + 1 - (len(current_row) + 1)):
                    continue
                candidates.append((x, new_mask))
            
            random.shuffle(candidates)
            
            for x, new_mask in candidates:
                used_mask |= new_mask
                current_row.append(x)
                
                if backtrack(row_idx, current_row):
                    return True
                
                current_row.pop()
                used_mask ^= new_mask
                
            return False
            
    if backtrack(0, [0]):
        return rows
    return None
