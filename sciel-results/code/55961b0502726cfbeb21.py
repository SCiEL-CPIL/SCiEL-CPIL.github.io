"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k):
    """Greedy construction."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _compress_all_rows(rows, deadline):
    """Iteratively compress all rows until no improvement or deadline."""
    rows = [list(r) for r in rows]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            
            old_row = rows[idx]
            n = len(old_row)
            r = list(old_row)
            
            # Try to reduce each position from right to left
            for pos in range(n - 1, 0, -1):
                if time.time() > deadline:
                    break
                
                old_val = r[pos]
                lower_bound = r[pos - 1] + 1
                
                # Try values from lower_bound to old_val - 1
                for new_val in range(lower_bound, old_val):
                    if time.time() > deadline:
                        break
                    
                    # Compute differences involving position pos
                    new_diffs = set()
                    for i in range(pos):
                        d = new_val - r[i]
                        if d <= 0:
                            break
                        new_diffs.add(d)
                    
                    # Get all diffs from the current row that involve position pos
                    old_row_diffs_involving_pos = set()
                    for i in range(pos):
                        old_row_diffs_involving_pos.add(old_val - r[i])
                    
                    # The "safe" used diffs are those not involving this position in the current row
                    safe_used = used - old_row_diffs_involving_pos
                    
                    if new_diffs & safe_used:
                        continue
                    
                    # Also check that new_diffs doesn't conflict with other diffs in this row
                    other_row_diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            if i == pos or j == pos:
                                continue
                            other_row_diffs.add(r[j] - r[i])
                    
                    if new_diffs & other_row_diffs:
                        continue
                    
                    # This value works
                    r[pos] = new_val
                    break
            
            if r != old_row:
                # Verify global validity
                test_rows = rows[:]
                test_rows[idx] = r
                if _valid(test_rows):
                    rows[idx] = r
                    improved = True
    
    return rows


def _local_search(rows, deadline, n_restarts=50):
    """Local search with simulated annealing to improve the solution."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Start from best or a perturbed version
        if restart == 0:
            current_rows = [list(r) for r in best_rows]
        else:
            current_rows = [list(r) for r in best_rows]
            # Perturb a random row
            idx = random.randint(0, len(current_rows) - 1)
            pos = random.randint(1, len(current_rows[idx]) - 1)
            current_rows[idx][pos] += random.randint(1, 20)
        
        current_scope = _scope(current_rows)
        temperature = max(1, (best_scope - current_scope) * 0.5)
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Pick a random row and position to modify
            idx = random.randint(0, len(current_rows) - 1)
            pos = random.randint(1, len(current_rows[idx]) - 1)
            
            old_val = current_rows[idx][pos]
            new_val = old_val + random.randint(-5, 5)
            
            if new_val <= 0 or new_val <= current_rows[idx][pos - 1]:
                continue
            if pos < len(current_rows[idx]) - 1 and new_val >= current_rows[idx][pos + 1]:
                continue
            
            current_rows[idx][pos] = new_val
            
            if _valid(current_rows):
                new_scope = _scope(current_rows)
                delta = new_scope - current_scope
                
                if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                else:
                    current_rows[idx][pos] = old_val
            
            temperature *= 0.995
        
        if time.time() > deadline:
            break
    
    return best_rows


def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Compress the greedy solution
    if time.time() < deadline - 30.0:
        compressed = _compress_all_rows(best_rows, deadline - 30.0)
        c_scope = _scope(compressed)
        if c_scope < best_scope:
            best_rows = compressed
            best_scope = c_scope
    
    # Phase 3: Local search with simulated annealing
    if time.time() < deadline - 10.0:
        ls_rows = _local_search(best_rows, deadline - 5.0, n_restarts=100)
        ls_scope = _scope(ls_rows)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    # Phase 4: Final compression pass
    if time.time() < deadline - 3.0:
        final_compressed = _compress_all_rows(best_rows, deadline - 2.0)
        final_scope = _scope(final_compressed)
        if final_scope < best_scope:
            best_rows = final_compressed
            best_scope = final_scope
    
    return best_rows

# EVOLVE-BLOCK-END
