import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Total differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Strategy:
    # 1. Get a good baseline via randomized greedy
    # 2. Use simulated annealing with row-regeneration moves to minimize scope
    # 3. The key insight: instead of element-level moves, regenerate entire rows
    #    which allows larger structural changes while maintaining validity
    
    # Step 1: Baseline
    baseline_rows = greedy_construct(n_rows, k, deadline)
    best_scope = max(max(row) for row in baseline_rows)
    best_rows = [row[:] for row in baseline_rows]
    
    # Step 2: Simulated annealing with row-regeneration
    rows = [row[:] for row in baseline_rows]
    current_scope = best_scope
    
    # Precompute used differences
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    used_diffs = get_used_diffs(rows)
    
    # SA parameters
    initial_temp = 50.0
    cooling_rate = 0.995
    min_temp = 0.5
    steps_per_temp = 3
    
    temp = initial_temp
    best_temp = initial_temp
    
    while temp > min_temp and time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Choose a row to regenerate
            row_idx = random.randint(0, n_rows - 1)
            
            # Remove its differences
            old_row = rows[row_idx]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            used_diffs -= old_diffs
            
            # Try to regenerate the row with a target max
            # Target: try to keep scope similar or slightly reduce
            target_max = current_scope + random.randint(-5, 3)
            target_max = max(target_max, k)  # minimum possible
            
            new_row = try_regenerate_row(k, target_max, used_diffs, deadline)
            
            if new_row is not None:
                new_scope = max(max(r) for r in rows[:row_idx] + [new_row] + rows[row_idx+1:])
                delta_scope = new_scope - current_scope
                
                # Accept if better, or with probability exp(-delta/temp)
                if delta_scope <= 0 or random.random() < pow(2.718281828, -delta_scope / temp):
                    rows[row_idx] = new_row
                    used_diffs |= set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            used_diffs.add(new_row[j] - new_row[i])
                    current_scope = new_scope
                    
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [row[:] for row in rows]
            else:
                # Restore old differences
                used_diffs |= old_diffs
        
        temp *= cooling_rate
    
    return best_rows


def greedy_construct(n_rows, k, deadline):
    """Randomized greedy to build initial rows."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if time.time() > deadline:
                break
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        if len(row) < k + 1:
            # Fallback: extend greedily
            x = row[-1] if row else 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    
    return rows


def try_regenerate_row(k, target_max, used_diffs, deadline):
    """
    Try to build a row of k+1 increasing integers starting at 0,
    with max entry <= target_max, such that all internal differences
    are not in used_diffs.
    """
    max_attempts = 20
    for _ in range(max_attempts):
        if time.time() > deadline:
            return None
        
        row = [0]
        x = 0
        success = True
        
        while len(row) < k + 1:
            # Find valid candidates
            last = row[-1]
            candidates = []
            for candidate in range(last + 1, target_max + 1):
                new_diffs = {candidate - a for a in row}
                if not (new_diffs & used_diffs):
                    # Check space: need at least (k+1 - len(row) - 1) more steps
                    remaining = k + 1 - len(row) - 1
                    if target_max - candidate >= remaining:
                        candidates.append(candidate)
            
            if not candidates:
                success = False
                break
            
            # Choose randomly from top candidates for diversity
            if len(candidates) > 10:
                # Weight toward smaller values to keep scope tight
                weights = [1.0 / (c - last + 1) for c in candidates]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                chosen = candidates[-1]
                for c, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        chosen = c
                        break
            else:
                chosen = random.choice(candidates)
            
            row.append(chosen)
            x = chosen
        
        if success and len(row) == k + 1 and row[-1] <= target_max:
            return row
    
    return None


# Alternative: a faster local search approach as backup
def local_search_improve(rows, n_rows, k, deadline):
    """Try to reduce scope by moving individual elements."""
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    current_scope = max(max(row) for row in rows)
    best_scope = current_scope
    best_rows = [row[:] for row in rows]
    
    temp = 10.0
    cooling = 0.99
    
    while temp > 0.1 and time.time() < deadline:
        # Pick a random row and a random position
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # not the first element (0)
        
        old_val = rows[row_idx][pos]
        
        # Remove this element's contribution to differences
        row = rows[row_idx]
        removed_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            diff = abs(row[pos] - row[i])
            if diff > 0:
                removed_diffs.add(diff)
        
        # Tentatively remove
        for d in removed_diffs:
            used_diffs.discard(d)
        
        # Try to find a new value
        low = row[pos-1] + 1 if pos > 0 else 1
        high = row[pos+1] - 1 if pos < k else current_scope
        
        if low > high:
            low = row[pos-1] + 1 if pos > 0 else 1
            high = current_scope
        
        new_val = None
        for _ in range(50):
            if low > high:
                break
            candidate = random.randint(low, high)
            if candidate == old_val:
                continue
            
            # Check if this creates conflicts
            conflict = False
            new_diffs = set()
            for i in range(len(row)):
                if i == pos:
                    continue
                diff = abs(candidate - row[i])
                if diff > 0:
                    if diff in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(diff)
            
            if not conflict:
                new_val = candidate
                break
        
        if new_val is not None:
            # Apply the move
            rows[row_idx][pos] = new_val
            used_diffs |= new_diffs
            
            new_scope = max(max(r) for r in rows)
            delta = new_scope - current_scope
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / temp):
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                rows[row_idx][pos] = old_val
                used_diffs -= new_diffs
                used_diffs |= removed_diffs
        else:
            # Revert
            used_diffs |= removed_diffs
        
        temp *= cooling
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Step 1: Baseline
    baseline_rows = greedy_construct(n_rows, k, deadline)
    best_scope = max(max(row) for row in baseline_rows)
    best_rows = [row[:] for row in baseline_rows]
    
    # Step 2: Row-regeneration SA
    rows = [row[:] for row in baseline_rows]
    current_scope = best_scope
    
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    temp = 30.0
    cooling_rate = 0.997
    min_temp = 1.0
    
    while temp > min_temp and time.time() < deadline:
        if time.time() > deadline:
            break
        
        row_idx = random.randint(0, n_rows - 1)
        old_row = rows[row_idx][:]
        
        # Remove old diffs
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        used_diffs -= old_diffs
        
        # Try regeneration with target near current scope
        target_max = current_scope + random.randint(-3, 2)
        target_max = max(target_max, k)
        
        new_row = try_regenerate_row(k, target_max, used_diffs, deadline)
        
        if new_row is not None:
            new_row_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_row_diffs.add(new_row[j] - new_row[i])
            
            new_scope = max(max(rows[i]) for i in range(n_rows) if i != row_idx)
            new_scope = max(new_scope, new_row[-1])
            delta = new_scope - current_scope
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.1)):
                rows[row_idx] = new_row
                used_diffs |= new_row_diffs
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                used_diffs |= old_diffs
        else:
            used_diffs |= old_diffs
        
        temp *= cooling_rate
    
    # Step 3: Local search polish
    if time.time() < deadline - 10:
        polished, polished_scope = local_search_improve(best_rows, n_rows, k, deadline)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    return best_rows
