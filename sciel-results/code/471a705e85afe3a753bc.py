"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_differences(rows):
    """Return the set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k):
    """Greedy initialization: build rows one by one with smallest possible marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _sa_optimize(rows, n_rows, k, deadline):
    """Simulated annealing to minimize scope (largest entry)."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    
    # Build initial difference set
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(row[j] - row[i])
    
    current_scope = _get_scope(rows)
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() < deadline:
        # Pick a random row and a random position to modify
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Choose a random element (not the first, which is always 0) to perturb
        if len(row) <= 1:
            continue
        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Try a random new value
        # Bias toward values near current scope to try to reduce it
        max_val = current_scope
        if random.random() < 0.7:
            # Try to reduce: pick a smaller value
            new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
        else:
            # Try a random value in a wider range
            new_val = random.randint(1, max_val + 10)
        
        if new_val == old_val:
            continue
        
        # Check if new value maintains strict increase
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue
        
        # Compute the differences that would change
        # Old differences involving position pos
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - old_val)
        
        # New differences involving position pos
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)
        
        # Check validity: new_diffs must not conflict with (used - old_diffs)
        remaining_used = used - old_diffs
        if new_diffs & remaining_used:
            T *= cooling
            continue
        
        # Accept the move
        row[pos] = new_val
        used = (used - old_diffs) | new_diffs
        
        new_scope = _get_scope(rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Revert
            row[pos] = old_val
            used = (used - new_diffs) | old_diffs
        
        T *= cooling
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with greedy solution
    rows = _greedy_init(n_rows, k)
    
    # Verify validity
    if not _is_valid(rows):
        # Shouldn't happen with greedy, but safety check
        return rows
    
    # Optimize with simulated annealing
    rows = _sa_optimize(rows, n_rows, k, deadline)
    
    # Final validation
    if not _is_valid(rows):
        # Fallback to greedy if SA produced invalid result
        return _greedy_init(n_rows, k)
    
    return rows
# EVOLVE-BLOCK-END
