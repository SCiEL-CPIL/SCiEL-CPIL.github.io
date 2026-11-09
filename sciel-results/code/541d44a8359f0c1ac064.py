"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _generate_valid_rows(k, max_val):
    """Generate all rows of size k+1 starting at 0, strictly increasing, max <= max_val,
    with distinct internal differences."""
    rows = []
    target_size = k + 1
    
    def backtrack(current, used_diffs):
        if len(current) == target_size:
            rows.append(list(current))
            return
        
        start = current[-1] + 1 if current else 1
        for x in range(start, max_val + 1):
            new_diffs = set()
            valid = True
            for a in current:
                d = x - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                current.append(x)
                backtrack(current, used_diffs | new_diffs)
                current.pop()
                
    backtrack([0], set())
    return rows

def _solve_packing(candidate_rows, n_rows, k):
    """Select n_rows disjoint rows from candidate_rows."""
    if len(candidate_rows) < n_rows:
        return None
        
    row_diffs = []
    for r in candidate_rows:
        diffs = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
        row_diffs.append(diffs)
        
    # Sort indices by max element of the row (ascending)
    indices = list(range(len(candidate_rows)))
    indices.sort(key=lambda i: candidate_rows[i][-1])
    
    selected = []
    used_diffs = set()
    
    def backtrack(idx, depth):
        if depth == n_rows:
            return True
        if idx >= len(indices):
            return False
            
        for i in range(idx, len(indices)):
            r_idx = indices[i]
            if not (row_diffs[r_idx] & used_diffs):
                selected.append(candidate_rows[r_idx])
                used_diffs |= row_diffs[r_idx]
                
                if backtrack(i + 1, depth + 1):
                    return True
                else:
                    used_diffs -= row_diffs[r_idx]
                    selected.pop()
                    
        return False

    if backtrack(0, 0):
        return selected
    return None

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    # Lower bound: n_rows * (k+1)k / 2
    lb = int(n_rows * k * (k + 1) / 2)
    
    # Search M from lb upwards
    # Limit search range to avoid timeout if lb is far from solution
    # For (7,5), lb=105, sol=112. Range 10 is safe.
    for M in range(lb, lb + 20):
        if time.time() - start_time > time_limit:
            break
            
        candidates = _generate_valid_rows(k, M)
        
        if not candidates:
            continue
            
        result = _solve_packing(candidates, n_rows, k)
        
        if result:
            return result
            
    # Fallback: Greedy
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

# EVOLVE-BLOCK-END
