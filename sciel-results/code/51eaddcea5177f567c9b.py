"""Difference triangle set (4,7) - greedy + local search."""
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130
    
    # Multiple greedy attempts with different strategies
    best_rows = None
    best_scope = float('inf')
    
    # Standard greedy
    rows = _greedy(n_rows, k)
    scope = rows[-1][-1]
    if scope < best_scope:
        best_scope = scope
        best_rows = rows
    
    # Try greedy with different row orderings (which row to build first)
    for attempt in range(50):
        if time.time() > deadline:
            break
        rows = _greedy_variant(n_rows, k, attempt)
        if rows is not None:
            scope = rows[-1][-1]
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Local search: try to reduce scope by modifying rows
    if best_rows is not None:
        improved = _local_search(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            scope = improved[-1][-1]
            if scope < best_scope:
                best_scope = scope
                best_rows = improved
    
    return best_rows


def _greedy(n_rows, k):
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


def _greedy_variant(n_rows, k, seed):
    """Greedy with slight randomization."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # With some probability, skip this and try next
                if rng.random() < 0.1 and x < 200:
                    continue
                row.append(x)
                used |= new_diffs
        if len(row) != k + 1:
            return None
        rows.append(row)
    return rows


def _local_search(rows, n_rows, k, start_time, deadline):
    """Try to reduce the scope by local modifications."""
    current_rows = [r[:] for r in rows]
    current_scope = max(r[-1] for r in current_rows)
    
    # Try to reduce the last element of rows
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n_rows):
            if current_rows[i][-1] <= 0:
                continue
            
            # Try to reduce the last element
            old_last = current_rows[i][-1]
            # Compute used differences excluding this row's contributions involving last element
            # This is complex; instead, try removing and re-adding with smaller value
            
            # Try each possible smaller value for the last element
            for new_last in range(old_last - 1, 0, -1):
                if time.time() > deadline:
                    break
                
                # Check if new_last is valid
                row = current_rows[i]
                new_diffs = {new_last - a for a in row[:-1]}
                
                # Check against all other rows' differences
                conflict = False
                for j in range(n_rows):
                    if j == i:
                        continue
                    rj = current_rows[j]
                    for a in rj:
                        for b in rj:
                            if a > b:
                                d = a - b
                                if d in new_diffs:
                                    conflict = True
                                    break
                            elif b > a:
                                d = b - a
                                if d in new_diffs:
                                    conflict = True
                                    break
                    if conflict:
                        break
                    # Also check within this row (excluding last element's diffs)
                if conflict:
                    continue
                
                # Also check within the same row (differences not involving last element)
                for a in row[:-1]:
                    for b in row[:-1]:
                        if a > b:
                            d = a - b
                            if d in new_diffs:
                                conflict = True
                                break
                if conflict:
                    continue
                
                # Valid! Update
                current_rows[i][-1] = new_last
                improved = True
                current_scope = max(r[-1] for r in current_rows)
                break
        
        if not improved:
            break
    
    if current_scope < max(r[-1] for r in rows):
        return current_rows
    return None
