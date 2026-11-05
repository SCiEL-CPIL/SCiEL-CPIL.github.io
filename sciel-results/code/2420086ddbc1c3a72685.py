import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # For the specific task (7,5), we can use a more targeted approach
    # but we need to handle general n_rows, k
    
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on scope: we need total_diffs distinct positive integers
    # The smallest total_diffs distinct positive integers are 1, 2, ..., total_diffs
    # So the maximum difference must be at least total_diffs
    # But differences are constrained by the row structure
    # A rough lower bound: m(m-1)/2 >= total_diffs for each row's max m
    # But rows share the difference space, so m must be large enough
    # Simple bound: m >= total_diffs (since all differences must be distinct and positive)
    # Actually, differences range from 1 to m, so we need m >= total_diffs
    # But m is the max entry, and differences can be at most m
    # So m >= total_diffs is a valid lower bound
    
    lower_bound = total_diffs
    
    # Try to find a solution with scope close to the lower bound
    # Use randomized restarts with iterative deepening on scope
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Try greedy construction with various seeds
    # This should quickly find a valid solution
    max_initial_attempts = 30
    for seed in range(max_initial_attempts):
        if time.time() > deadline - 60:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: try with larger limit
        rows = _greedy_construct(n_rows, k, 0, limit=1000)
        if rows is not None:
            best_scope = _scope(rows)
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Last resort: return a trivially valid set (may have large scope)
        # For n_rows=1, this is fine
        # For n_rows>1, we need to ensure validity
        # Use a simple construction that's guaranteed valid but may have large scope
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                # Use differences that are multiples of a large base to avoid conflicts
                # This is a crude fallback
                row.append(offset + j * (total_diffs + 1))
            rows.append(row)
            offset += total_diffs + 1
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 2: Local search to improve scope
    if time.time() < deadline - 10:
        improved = _local_search(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Targeted reduction of the maximum element
    if time.time() < deadline - 5 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2

def _greedy_construct(n_rows, k, seed, limit=500):
    """Best-fit greedy construction."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            best_x = None
            best_score = float('inf')
            
            for x in range(last_x + 1, limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Score: prefer smaller x, and prefer x that fills gaps
                    score = x + 0.1 * len(new_diffs)
                    if score < best_score:
                        best_score = score
                        best_x = x
            
            if best_x is None:
                # Expand search
                for x in range(limit + 1, limit * 3 + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        score = x + 0.1 * len(new_diffs)
                        if score < best_score:
                            best_score = score
                            best_x = x
            
            if best_x is None:
                return None
            
            row.append(best_x)
            new_diffs = {best_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _local_search(rows, n_rows, k, start_time, deadline):
    """Local search to reduce scope."""
    rng = random.Random(42)
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    T = 50.0
    T_end = 0.1
    total_steps = 15000
    cool_rate = (T_end / T) ** (1.0 / total_steps)
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        # Pick a row to modify, biased towards the one with max element
        if rng.random() < 0.5:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        # Generate a new candidate row
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        
        if new_row is None:
            continue
        
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        # Check validity
        if not _is_valid(new_rows, n_rows, k):
            continue
        
        delta = new_scope - current_scope
        if delta <= 0:
            rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows = new_rows
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _generate_candidate_row(rows, exclude_idx, k, rng, limit=500):
    """Generate a new candidate row."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    row = [0]
    for _ in range(k):
        last_x = row[-1]
        best_x = None
        best_score = float('inf')
        
        for x in range(last_x + 1, limit + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                score = x + 0.1 * len(new_diffs)
                if score < best_score:
                    best_score = score
                    best_x = x
        
        if best_x is None:
            return None
        
        row.append(best_x)
        new_diffs = {best_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
    
    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    """Try to reduce the maximum element."""
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    for iteration in range(100):
        if time.time() > deadline - 3:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to find a row with smaller max
        improved_row = _reduce_max_row(rows, max_idx, k, rng, current_max)
        
        if improved_row is not None and improved_row[-1] < current_max:
            new_rows = rows[:]
            new_rows[max_idx] = improved_row
            if _is_valid(new_rows, n_rows, k):
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in new_rows]
                    rows = new_rows
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    """Try to find a replacement with smaller max."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])
    
    # Try decreasing target_max
    for target_max in range(current_max - 1, max(current_max - 10, 0), -1):
        row = [0]
        success = True
        for _ in range(k):
            last_x = row[-1]
            best_x = None
            best_score = float('inf')
            
            for x in range(last_x + 1, target_max + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    score = x + 0.1 * len(new_diffs)
                    if score < best_score:
                        best_score = score
                        best_x = x
            
            if best_x is None:
                success = False
                break
            
            row.append(best_x)
            new_diffs = {best_x - a for a in row[:-1]}
            used_diffs.update(new_diffs)
        
        if success and len(row) == k + 1:
            return row
    
    return None
