"""Difference triangle set construction via randomized local search."""
import time
import random
import sys

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    
    All positive differences a_j - a_i (i < j) must be distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    
    # Lower bound on max value: we need total_diffs distinct positive integers
    # from 1..M, so M >= total_diffs
    lb = total_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    # Helper: check if a set of rows is valid
    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    # Helper: get scope (max entry) of rows
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    # Helper: greedy construction with randomization
    def greedy_random(seed_offset=0, bias=0.0):
        random.seed(seed_offset)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # With some probability, skip ahead
                if bias > 0 and random.random() < bias:
                    x += random.randint(1, 5)
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Helper: local search - try to reduce scope by modifying rows
    def local_search(rows, max_iters=5000):
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        scope = get_scope(rows)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try to reduce the largest element in each row
            for ri in range(n_rows):
                row = rows[ri]
                last = row[-1]
                # Try removing last element and finding a smaller one
                # First, remove all differences involving last
                old_diffs = set()
                for a in row[:-1]:
                    d = last - a
                    old_diffs.add(d)
                    used.discard(d)
                
                # Try values from last-1 down to row[-2]+1
                found = False
                for new_last in range(last - 1, row[-2], -1):
                    new_diffs = {new_last - a for a in row[:-1]}
                    if not new_diffs & used:
                        # Also need new_diffs to not overlap with each other (they won't since row is increasing)
                        row[-1] = new_last
                        used |= new_diffs
                        scope = min(scope, new_last)
                        improved = True
                        found = True
                        break
                
                if not found:
                    # Restore
                    row[-1] = last
                    used |= old_diffs
        
        return rows, scope
    
    # Helper: more aggressive local search - can move any element
    def local_search_full(rows, max_iters=2000):
        rows = [list(r) for r in rows]
        
        def rebuild_used():
            used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])
            return used
        
        used = rebuild_used()
        scope = get_scope(rows)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for ri in range(n_rows):
                row = rows[ri]
                for pos in range(1, len(row)):  # skip position 0 (always 0)
                    old_val = row[pos]
                    # Remove differences involving this element
                    old_diffs = set()
                    for i in range(pos):
                        d = old_val - row[i]
                        old_diffs.add(d)
                        used.discard(d)
                    for j in range(pos + 1, len(row)):
                        d = row[j] - old_val
                        old_diffs.add(d)
                        used.discard(d)
                    
                    # Try to find a smaller valid value
                    low = row[pos-1] + 1
                    high = old_val - 1
                    found = False
                    for new_val in range(high, low - 1, -1):
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, len(row)):
                            new_diffs.add(row[j] - new_val)
                        if not new_diffs & used:
                            row[pos] = new_val
                            used |= new_diffs
                            if new_val < scope:
                                scope = new_val
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        row[pos] = old_val
                        used |= old_diffs
        
        return rows, scope
    
    # Initial greedy
    t0 = time.time()
    rows = greedy_random(42, 0.0)
    if is_valid(rows):
        best_rows = [list(r) for r in rows]
        best_scope = get_scope(rows)
    
    # Try local search on the greedy result
    if time.time() < deadline - 10:
        rows_ls, scope_ls = local_search(rows)
        if is_valid(rows_ls):
            if scope_ls < best_scope:
                best_rows = [list(r) for r in rows_ls]
                best_scope = scope_ls
    
    # Try full local search
    if time.time() < deadline - 10:
        rows_full, scope_full = local_search_full(best_rows, max_iters=1000)
        if is_valid(rows_full):
            if scope_full < best_scope:
                best_rows = [list(r) for r in rows_full]
                best_scope = scope_full
    
    # Try multiple random greedy starts with local search
    seed = 1000
    while time.time() < deadline - 15:
        seed += 1
        bias = random.random() * 0.3
        rows_r = greedy_random(seed, bias)
        if not is_valid(rows_r):
            continue
        scope_r = get_scope(rows_r)
        if scope_r >= best_scope:
            # Still try local search if close
            if scope_r - best_scope > 20:
                continue
        rows_ls, scope_ls = local_search(rows_r, max_iters=2000)
        if is_valid(rows_ls):
            if scope_ls < best_scope:
                best_rows = [list(r) for r in rows_ls]
                best_scope = scope_ls
                # Try full local search on improved result
                if time.time() < deadline - 10:
                    rows_full, scope_full = local_search_full(best_rows, max_iters=500)
                    if is_valid(rows_full):
                        if scope_full < best_scope:
                            best_rows = [list(r) for r in rows_full]
                            best_scope = scope_full
    
    # Final check
    if best_rows is None:
        # Fallback to pure greedy
        rows = greedy_random(42, 0.0)
        best_rows = rows
        best_scope = get_scope(rows)
    
    return best_rows
