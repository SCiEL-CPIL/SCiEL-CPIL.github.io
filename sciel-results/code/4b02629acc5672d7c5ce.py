"""Combined program for difference triangle set (7,5)."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 115.0
    
    # Start with greedy construction as baseline
    best_rows = _greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # Try randomized greedy with efficient local search improvements
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        rows = _randomized_greedy(n_rows, k)
        if rows is None:
            continue
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        
        # Try efficient local search to reduce scope
        improved = _local_search(rows, n_rows, k, min(time.time() + 8.0, deadline - 5.0))
        if improved is not None:
            imp_scope = max(max(r) for r in improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = [list(r) for r in improved]
        
        # Occasionally try a more aggressive multi-start
        if attempts % 5 == 0 and time.time() < deadline - 20.0:
            for _ in range(3):
                if time.time() > deadline - 10.0:
                    break
                rows2 = _randomized_greedy(n_rows, k)
                if rows2 is None:
                    continue
                scope2 = max(max(r) for r in rows2)
                if scope2 < best_scope:
                    best_scope = scope2
                    best_rows = [list(r) for r in rows2]
                improved2 = _local_search(rows2, n_rows, k, min(time.time() + 5.0, deadline - 10.0))
                if improved2 is not None:
                    imp_scope2 = max(max(r) for r in improved2)
                    if imp_scope2 < best_scope:
                        best_scope = imp_scope2
                        best_rows = [list(r) for r in improved2]
    
    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
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

def _randomized_greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            # Try candidates in a window near the minimum
            candidates = []
            for x_try in range(x + 1, x + 30):
                new = {x_try - a for a in row}
                if not new & used:
                    candidates.append(x_try)
                if len(candidates) >= 5:
                    break
            if not candidates:
                # Fallback to linear search
                x_try = x + 1
                while True:
                    new = {x_try - a for a in row}
                    if not new & used:
                        candidates.append(x_try)
                        break
                    x_try += 1
            
            # Pick randomly from early candidates for diversity
            if len(candidates) > 1:
                idx = random.randint(0, min(2, len(candidates) - 1))
                x = candidates[idx]
            else:
                x = candidates[0]
            
            new = {x - a for a in row}
            row.append(x)
            used |= new
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline):
    """Efficient local search: maintain global diff set, only check affected diffs."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)
    
    # Build global used differences set
    used_diffs = set()
    for r in best:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                used_diffs.add(r[j] - r[i])
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle row order to avoid bias
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for i in row_order:
            if time.time() > deadline:
                break
            row = best[i]
            # Try to reduce each element from largest to smallest
            for j in range(len(row) - 1, 0, -1):
                old_val = row[j]
                if old_val <= row[j-1] + 1:
                    continue
                
                # Find the minimum valid replacement
                # Must be > row[j-1] and < old_val
                # Check candidates from row[j-1]+1 up to old_val-1
                found = False
                for new_val in range(row[j-1] + 1, old_val):
                    # Check if new_val creates valid differences
                    # Diffs with previous elements in same row: new_val - row[a] for a < j
                    # Diffs with next elements in same row: row[b] - new_val for b > j
                    
                    # Remove old diffs involving position j from used_diffs
                    old_diffs = set()
                    for a in range(j):
                        old_diffs.add(old_val - row[a])
                    for b in range(j + 1, len(row)):
                        old_diffs.add(row[b] - old_val)
                    
                    # Check new diffs
                    new_diffs = set()
                    valid = True
                    for a in range(j):
                        d = new_val - row[a]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    for b in range(j + 1, len(row)):
                        d = row[b] - new_val
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    # Check against other rows' diffs (used_diffs minus old_diffs)
                    # A diff is blocked if it's in used_diffs but NOT in old_diffs
                    blocked = False
                    for d in new_diffs:
                        if d in used_diffs and d not in old_diffs:
                            blocked = True
                            break
                    
                    if not blocked:
                        # Apply the change
                        row[j] = new_val
                        used_diffs -= old_diffs
                        used_diffs |= new_diffs
                        new_scope = max(max(r) for r in best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        found = True
                        break
                
                if found:
                    continue  # Move to next position after successful change
    
        if not improved:
            break
    
    return best
# EVOLVE-BLOCK-END
