import time

def _greedy_construct(n_rows, k):
    """Fallback greedy construction."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _generate_rulers(k, M):
    """Generate all Golomb rulers of length k+1 with max element <= M."""
    rulers = []
    marks = [0]
    diffs = set()
    
    def gen(idx, current_max):
        if idx == k:
            # Store as (max_elem, diff_mask, marks_copy)
            diff_mask = 0
            for d in diffs:
                diff_mask |= (1 << d)
            rulers.append((marks[-1], diff_mask, marks[:]))
            return
        
        # Next mark must be > current_max
        for next_mark in range(current_max + 1, M + 1):
            new_diffs = {next_mark - m for m in marks}
            if new_diffs & diffs:
                continue
            
            marks.append(next_mark)
            diffs.update(new_diffs)
            
            gen(idx + 1, next_mark)
            
            marks.pop()
            diffs.difference_update(new_diffs)

    gen(1, 0)
    return rulers

def _find_packing(rulers, n_rows, time_limit=5.0):
    """Find n_rows rulers with disjoint difference sets."""
    if not rulers:
        return None
    
    # Sort by max element to find small solutions first
    rulers.sort(key=lambda x: x[0])
    
    start_time = time.time()
    chosen = []
    
    def search(i, used_mask):
        nonlocal chosen, start_time
        
        if len(chosen) == n_rows:
            return True
        
        # Pruning: not enough rulers left
        if len(rulers) - i < n_rows - len(chosen):
            return False
            
        # Time check
        if time.time() - start_time > time_limit:
            return False
            
        for j in range(i, len(rulers)):
            r = rulers[j]
            # Check disjointness
            if r[1] & used_mask:
                continue
                
            chosen.append(r)
            if search(j + 1, used_mask | r[1]):
                return True
            chosen.pop()
            
        return False

    if search(0, 0):
        return [r[2] for r in chosen]
    return None

def construct(n_rows, k):
    # Lower bound on max element
    num_diffs = n_rows * k * (k + 1) // 2
    lb = num_diffs
    
    # Try exact search for small instances
    # If n_rows * k is small, exact search is feasible
    if n_rows * k <= 50: # Heuristic threshold
        for M in range(lb, lb + 30):
            rulers = _generate_rulers(k, M)
            if not rulers:
                continue
            
            result = _find_packing(rulers, n_rows)
            if result:
                return result
        
    # Fallback to greedy
    return _greedy_construct(n_rows, k)
