import time
import random
import sys

def construct_greedy_row(forbidden, k, limit=None):
    """
    Construct a row of k+1 integers starting at 0, strictly increasing,
    such that all internal differences are not in 'forbidden'.
    Minimizes the last element.
    If limit is provided, stops if last element exceeds limit.
    Returns the row list or None if not found.
    """
    row = [0]
    used_in_row = set()
    current_max = 0
    
    # We need to find k-1 more numbers
    # The next number x must satisfy:
    # 1. x > current_max
    # 2. {x - a for a in row} is disjoint from forbidden
    # 3. {x - a for a in row} is disjoint from used_in_row (internal consistency, which is guaranteed if we pick x > current_max and row is strictly increasing? 
    #    Wait. If row = [0, 2, 5], diffs are 2, 5, 3.
    #    If we add 7: diffs 7, 5, 2. 5 and 2 are already in used_in_row. 
    #    The condition is "all positive differences ... are distinct over all rows".
    #    It does NOT say differences within a row must be distinct from each other?
    #    "all differences between two entries of the same row are distinct over all rows"
    #    This phrasing usually means the set of all differences from all rows is a set (no duplicates).
    #    So yes, within a row, differences must also be distinct.
    #    A set of integers with distinct pairwise differences is a Golomb Ruler.
    #    So yes, we must ensure internal diffs are distinct too.
    
    # So, for a new candidate x:
    # new_diffs = {x - a for a in row}
    # Condition: (new_diffs & forbidden) is empty
    # Condition: (new_diffs & used_in_row) is empty
    
    while len(row) < k + 1:
        x = current_max + 1
        found = False
        while True:
            if limit is not None and x > limit:
                return None
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden) and not (new_diffs & used_in_row):
                row.append(x)
                used_in_row |= new_diffs
                current_max = x
                found = True
                break
            x += 1
        if not found:
            return None
    return row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial Greedy Construction
    rows = []
    global_forbidden = set()
    
    for _ in range(n_rows):
        row = construct_greedy_row(global_forbidden, k)
        if row is None:
            # Should not happen for reasonable n, k
            return []
        rows.append(row)
        # Add row's diffs to global forbidden
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                global_forbidden.add(row[j] - row[i])
                
    best_scope = max(max(r) for r in rows)
    
    # Precompute diff sets for each row for fast lookup
    row_diffs = []
    for r in rows:
        d = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d.add(r[j] - r[i])
        row_diffs.append(d)
        
    no_improvement_time = 0.0
    last_improvement = start_time
    
    while time.time() < deadline:
        # Pick a random row to try to shrink
        idx = random.randint(0, n_rows - 1)
        
        # Calculate forbidden set from all OTHER rows
        # Efficient way: total_diffs - row_diffs[idx]
        # But sets don't support subtraction of elements if duplicates exist across rows?
        # The problem states differences are distinct OVER ALL ROWS.
        # So the global set of differences is the union, and it's a set (no duplicates).
        # So forbidden = global_diffs - row_diffs[idx]
        
        # To avoid rebuilding global_diffs every time, we can maintain it.
        # However, if we change a row, global_diffs changes.
        # Let's just recompute forbidden for the specific idx.
        # n_rows is likely small (e.g. 7). k is small (e.g. 5 or 6).
        # So row_diffs are small sets.
        
        forbidden = set()
        for j in range(n_rows):
            if j != idx:
                forbidden.update(row_diffs[j])
        
        current_max = rows[idx][-1]
        
        # Try to find a better row
        # We want a row with max < current_max
        # If current_max is the global best scope, we want max < best_scope
        # But even if current_max > best_scope, shrinking it might not help global best
        # unless it allows other rows to shrink? No, shrinking one row frees up differences,
        # potentially allowing others to shrink.
        
        # Let's try to find a row with max < current_max
        new_row = construct_greedy_row(forbidden, k, limit=current_max - 1)
        
        if new_row is not None:
            new_max = new_row[-1]
            # Update the row
            # Remove old diffs from global? We don't maintain a global set explicitly in the loop, 
            # we recompute forbidden on the fly.
            
            # Calculate new diffs
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
                    
            # Check validity (should be valid by construction)
            # Update state
            rows[idx] = new_row
            row_diffs[idx] = new_diffs
            
            new_global_scope = max(max(r) for r in rows)
            if new_global_scope < best_scope:
                best_scope = new_global_scope
                last_improvement = time.time()
            # Even if global scope doesn't improve, the row is smaller, 
            # which might help future shrinks of other rows.
            
    return rows
