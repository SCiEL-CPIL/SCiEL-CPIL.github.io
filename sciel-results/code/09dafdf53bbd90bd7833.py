import time
import random
import sys
from itertools import combinations

def _compute_diffs(row):
    return frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

def _is_valid(rows):
    all_diffs = set()
    for row in rows:
        diffs = _compute_diffs(row)
        if diffs & all_diffs:
            return False
        all_diffs |= diffs
    return True

def _get_scope(rows):
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, deadline):
    """
    Generate an initial set of rows using a greedy approach.
    """
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        if time.time() > deadline:
            break
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, 200):
                new_diffs = frozenset(x - a for a in row)
                if new_diffs & used_diffs:
                    continue
                row.append(x)
                used_diffs |= new_diffs
                current_max = x
                found = True
                break
            if not found:
                return None
        rows.append(row)
    return rows

def _rebuild_row(rows, idx, k, deadline):
    """
    Try to rebuild a single row to reduce its scope.
    """
    other_diffs = set()
    for i, row in enumerate(rows):
        if i == idx:
            continue
        other_diffs |= _compute_diffs(row)
    
    old_row = rows[idx]
    old_max = max(old_row)
    
    # Try to find a row with a smaller max value
    for new_max in range(old_max - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Backtracking search for a row with max value <= new_max
        def backtrack(pos, current_row, used):
            if time.time() > deadline:
                return None
            if pos == k + 1:
                return current_row[:]
            
            start = current_row[-1] + 1 if pos > 0 else 0
            for val in range(start, new_max + 1):
                new_diffs = frozenset(val - a for a in current_row)
                if new_diffs & used:
                    continue
                if new_diffs & other_diffs:
                    continue
                
                current_row.append(val)
                used |= new_diffs
                
                result = backtrack(pos + 1, current_row, used)
                if result is not None:
                    return result
                
                current_row.pop()
                used -= new_diffs
            
            return None
        
        result = backtrack(0, [0], set())
        if result is not None:
            return result
    
    return None

def _sa_optimize(rows, k, deadline):
    """
    Simulated Annealing optimization to reduce the scope.
    """
    current_rows = [row[:] for row in rows]
    current_scope = _get_scope(current_rows)
    best_rows = [row[:] for row in current_rows]
    best_scope = current_scope
    
    t = 10.0
    cooling = 0.995
    
    while time.time() < deadline and t > 0.1:
        # Pick a random row to modify
        idx = random.randint(0, len(current_rows) - 1)
        old_row = current_rows[idx][:]
        
        # Generate a new random row
        new_row = [0]
        current_max = 0
        used_diffs = set()
        other_diffs = set()
        for i, row in enumerate(current_rows):
            if i == idx:
                continue
            other_diffs |= _compute_diffs(row)
        
        while len(new_row) < k + 1:
            found = False
            for x in range(current_max + 1, 200):
                new_diffs = frozenset(x - a for a in new_row)
                if new_diffs & used_diffs or new_diffs & other_diffs:
                    continue
                new_row.append(x)
                used_diffs |= new_diffs
                current_max = x
                found = True
                break
            if not found:
                break
        
        if len(new_row) < k + 1:
            t *= cooling
            continue
        
        new_scope = max(current_scope, max(new_row))
        
        # Accept if better or with probability exp(-delta/t)
        delta = new_scope - current_scope
        if delta < 0 or random.random() < pow(2.718281828, -delta / t):
            current_rows[idx] = new_row
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in current_rows]
        
        t *= cooling
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145
    
    # Get initial solution
    rows = _greedy_init(n_rows, k, deadline)
    if rows is None:
        # Fallback to a simple construction
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                for x in range(current_max + 1, 300):
                    new_diffs = frozenset(x - a for a in row)
                    if new_diffs & used_diffs:
                        continue
                    row.append(x)
                    used_diffs |= new_diffs
                    current_max = x
                    break
            rows.append(row)
    
    # Optimize using simulated annealing
    rows = _sa_optimize(rows, k, deadline)
    
    return rows
