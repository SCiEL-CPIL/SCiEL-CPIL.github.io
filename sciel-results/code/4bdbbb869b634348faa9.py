"""Improved program for difference triangle set (4,7)."""
import sys
sys.setrecursionlimit(100000)

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    min_scope = total_diffs
    
    # Try a range of scopes
    for scope in range(min_scope, min_scope + 30):
        result = _backtrack_construct(n_rows, k, scope)
        if result is not None:
            return result
    
    # Fallback to greedy
    return _greedy(n_rows, k)


def _backtrack_construct(n_rows, k, scope):
    used = set()
    rows = []
    
    def backtrack(row_idx, current_row, pos_idx):
        if row_idx == n_rows:
            return True
        if pos_idx == k + 1:
            rows.append(current_row[:])
            return backtrack(row_idx + 1, [0], 1)
        
        start = current_row[-1] + 1
        # Try positions in order
        for pos in range(start, scope + 1):
            # Compute new differences
            new_diffs = []
            ok = True
            for prev in current_row:
                d = pos - prev
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue
            
            # Forward check: ensure we can still complete the row
            # Remaining positions: (k+1) - pos_idx
            # Each will add at least 1 new difference
            # This is a weak check, skip for now
            
            current_row.append(pos)
            for d in new_diffs:
                used.add(d)
            
            if backtrack(row_idx, current_row, pos_idx + 1):
                return True
            
            current_row.pop()
            for d in new_diffs:
                used.discard(d)
        
        return False
    
    if backtrack(0, [0], 1):
        return rows
    return None


def _greedy(n_rows, k):
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
# EVOLVE-BLOCK-END
