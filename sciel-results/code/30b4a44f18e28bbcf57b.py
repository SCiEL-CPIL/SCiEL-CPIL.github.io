"""Difference triangle set (4,7): difference-packing with randomized search and local improvement."""
import time
import sys
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    
    Strategy: 
    1. Use randomized greedy with multiple restarts to find initial solutions
    2. Apply local compaction to reduce scope
    3. Use simulated annealing for further optimization
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: scope >= total_diffs
    lb = total_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Randomized greedy with multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Random seed for diversity
        random.seed(restart * 7919 + 13)
        
        rows = _randomized_greedy(n_rows, k, lb)
        if rows is None:
            continue
        
        scope = max(max(row) for row in rows)
        
        # Try to compact
        rows = _local_compact(rows, n_rows, k, deadline)
        scope = max(max(row) for row in rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _greedy_construct(n_rows, k)
        return best_rows
    
    # Phase 2: Simulated annealing on the best solution
    if time.time() < deadline - 15.0:
        rows = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        scope = max(max(row) for row in rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    # Phase 3: Final local compaction
    best_rows = _local_compact(best_rows, n_rows, k, deadline)
    
    return best_rows


def _randomized_greedy(n_rows, k, min_scope):
    """
    Randomized greedy construction.
    Build rows one at a time, choosing values with some randomization.
    """
    used_diffs = set()
    rows = []
    
    for r in range(n_rows):
        row = [0]
        remaining = k
        
        # Choose values for positions 1..k
        while len(row) < k + 1:
            # Determine valid range for next value
            min_val = row[-1] + 1
            max_val = min_scope + r * 10  # allow some slack
            
            # Collect valid candidates
            candidates = []
            for val in range(min_val, max_val + 1):
                new_diffs = {val - a for a in row}
                if not new_diffs & used_diffs:
                    candidates.append(val)
            
            if not candidates:
                # Try larger range
                max_val = min_scope + r * 10 + 50
                for val in range(min_val, max_val + 1):
                    new_diffs = {val - a for a in row}
                    if not new_diffs & used_diffs:
                        candidates.append(val)
            
            if not candidates:
                return None
            
            # Choose randomly from top candidates (with bias toward smaller values)
            # Use exponential decay: smaller values more likely
            if len(candidates) > 1:
                weights = [1.0 / (1.0 + i * 0.1) for i in range(len(candidates))]
                total_w = sum(weights)
                rand_val = random.random() * total_w
                cum = 0
                chosen_idx = len(candidates) - 1
                for i, w in enumerate(weights):
                    cum += w
                    if rand_val <= cum:
                        chosen_idx = i
                        break
                val = candidates[chosen_idx]
            else:
                val = candidates[0]
            
            row.append(val)
            new_diffs = {val - a for a in row[:-1]}
            used_diffs |= new_diffs
        
        rows.append(row)
    
    return rows


def _local_compact(rows, n_rows, k, deadline):
    """
    Try to reduce the scope by compacting each row.
    For each row, try to shift values down where possible.
    """
    if rows is None:
        return None
    
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    # Try to compact each row independently
    for r in range(n_rows):
        if time.time() > deadline - 5.0:
            break
        
        row = rows[r]
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                row_diffs.add(row[j] - row[i])
        
        other_diffs = used_diffs - row_diffs
        
        # Try to shift each position down
        changed = True
        while changed:
            changed = False
            for pos in range(1, len(row)):
                if time.time() > deadline - 5.0:
                    break
                
                current = row[pos]
                # Try to reduce this position
                for new_val in range(row[pos-1] + 1, current):
                    # Check if new value is valid
                    new_diffs = set()
                    for i in range(pos):
                        d = new_val - row[i]
                        new_diffs.add(d)
                    
                    # Check against other diffs and existing row diffs (excluding those involving this position)
                    old_diffs = set()
                    for i in range(pos):
                        d = current - row[i]
                        old_diffs.add(d)
                    
                    # New diffs must not conflict with other_diffs or remaining row_diffs
                    remaining_row_diffs = row_diffs - old_diffs
                    if new_diffs & other_diffs or new_diffs & remaining_row_diffs:
                        continue
                    
                    # Also check that new_diffs don't conflict with each other (they won't since row is increasing)
                    row[pos] = new_val
                    row_diffs = row_diffs - old_diffs | new_diffs
                    changed = True
                    break
        
        # Update used_diffs
        used_diffs = used_diffs - row_diffs
        new_row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                new_row_diffs.add(row[j] - row[i])
        used_diffs |= new_row_diffs
    
    return rows


def _simulated_annealing(rows, n_rows, k, start_time, deadline):
    """
    Simulated annealing to further reduce scope.
    """
    if rows is None:
        return None
    
    current_rows = [list(r) for r in rows]
    current_scope = max(max(row) for row in current_rows)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    temperature = 10.0
    cooling_rate = 0.95
    
    # Compute initial used diffs
    used_diffs = set()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    max_iterations = 5000
    iteration = 0
    
    while time.time() < deadline - 10.0 and iteration < max_iterations:
        iteration += 1
        
        # Choose a random row and position to perturb
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        # Try to change this position
        old_val = current_rows[r][pos]
        
        # Generate new value: try nearby values
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Ensure validity: must be > previous and < next (if exists)
        if new_val <= current_rows[r][pos - 1]:
            new_val = current_rows[r][pos - 1] + 1
        if pos < k and new_val >= current_rows[r][pos + 1]:
            new_val = current_rows[r][pos + 1] - 1
            if new_val <= current_rows[r][pos - 1]:
                continue
        
        # Compute new diffs for this row
        old_row_diffs = set()
        for i in range(len(current_rows[r])):
            for j in range(i + 1, len(current_rows[r])):
                old_row_diffs.add(current_rows[r][j] - current_rows[r][i])
        
        # Temporarily update
        current_rows[r][pos] = new_val
        
        new_row_diffs = set()
        for i in range(len(current_rows[r])):
            for j in range(i + 1, len(current_rows[r])):
                new_row_diffs.add(current_rows[r][j] - current_rows[r][i])
        
        # Check for conflicts
        other_diffs = used_diffs - old_row_diffs
        if new_row_diffs & other_diffs:
            # Conflict: revert
            current_rows[r][pos] = old_val
            continue
        
        # Accept or reject based on temperature
        new_scope = max(max(row) for row in current_rows)
        delta_scope = new_scope - current_scope
        
        if delta_scope <= 0:
            # Better or equal: accept
            current_scope = new_scope
            used_diffs = other_diffs | new_row_diffs
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Worse: accept with probability exp(-delta_scope / temperature)
            import math
            prob = math.exp(-delta_scope / max(temperature, 0.01))
            if random.random() < prob:
                current_scope = new_scope
                used_diffs = other_diffs | new_row_diffs
            else:
                # Revert
                current_rows[r][pos] = old_val
        
        # Cool down
        temperature *= cooling_rate
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Greedy fallback construction."""
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
