import time
import random
import copy

def _greedy_construct(n_rows, k):
    """
    Greedy construction of n_rows Golomb rulers with disjoint differences.
    Returns a list of rows.
    """
    size = k + 1
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < size:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                x += 1
            else:
                x += 1
        rows.append(row)
    return rows

def _get_diffs(row):
    """
    Get the set of differences for a row.
    """
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _get_diffs_involving_index(row, j):
    """
    Get the set of differences involving index j in the row.
    """
    diffs = set()
    for i in range(len(row)):
        if i == j:
            continue
        if i < j:
            diffs.add(row[j] - row[i])
        else:
            diffs.add(row[i] - row[j])
    return diffs

def construct(n_rows, k):
    deadline = time.time() + 110.0 # Safety margin for 120s limit
    
    # 1. Initial Solution
    current_rows = _greedy_construct(n_rows, k)
    best_rows = copy.deepcopy(current_rows)
    
    size = k + 1
    
    # Precompute diff counts
    diff_counts = {}
    for row in current_rows:
        for d in _get_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
    def get_conflicts(counts):
        return sum(c * (c - 1) // 2 for c in counts.values())
        
    current_conflicts = get_conflicts(diff_counts)
    current_max = max(max(row) for row in current_rows)
    best_max = current_max
    best_conflicts = current_conflicts
    
    # If initial is valid, best is set.
    if current_conflicts == 0:
        best_rows = copy.deepcopy(current_rows)
        best_max = current_max
    
    # Simulated Annealing / Local Search
    T = 10.0
    cooling = 0.999
    
    def calc_cost(conflicts, max_val):
        # We prioritize validity, then minimization of max
        # If conflicts > 0, cost is huge
        return max_val + 10000 * conflicts

    current_cost = calc_cost(current_conflicts, current_max)
    
    while time.time() < deadline:
        # Pick random row and index (1 to size-1)
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, size - 1)
        
        v = current_rows[r][j]
        
        # Determine bounds for new_v
        lower = current_rows[r][j-1] + 1
        upper = current_rows[r][j+1] - 1 if j < size - 1 else best_max + 5 # Allow slight expansion if stuck?
        # To minimize max, we should ideally restrict upper to current_max or best_max.
        # But sometimes we need to move out to move in.
        # Let's allow upper to be max(v + 10, best_max)
        
        if upper < lower:
            continue
            
        # Random new value
        # Bias towards smaller values to reduce max
        # new_v = random.randint(lower, min(upper, best_max))
        new_v = random.randint(lower, upper)
        
        if new_v == v:
            continue
            
        # Calculate old diffs involving j
        old_diffs = _get_diffs_involving_index(current_rows[r], j)
        
        # Tentatively apply
        current_rows[r][j] = new_v
        new_diffs = _get_diffs_involving_index(current_rows[r], j)
        
        # Update diff_counts
        # Remove old
        for d in old_diffs:
            c = diff_counts[d]
            diff_counts[d] = c - 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Add new
        for d in new_diffs:
            c = diff_counts.get(d, 0)
            diff_counts[d] = c + 1
            
        # Calculate new conflicts
        # Instead of summing all, we can track delta?
        # For simplicity and safety, recompute conflicts from diff_counts.
        # diff_counts has at most ~200 keys. Sum is fast.
        new_conflicts = get_conflicts(diff_counts)
        
        # Calculate new max
        new_row_max = max(current_rows[r])
        # Global max: if new_row_max > current_max, it's new_row_max.
        # If new_row_max < current_max, we need to check if it was the unique max.
        # Since n_rows is small, just compute max of all rows?
        # Or keep a max_heap?
        # Simple: compute max. 7 rows * 6 ints is trivial.
        new_global_max = max(max(row) for row in current_rows)
        
        new_cost = calc_cost(new_conflicts, new_global_max)
        
        delta_cost = new_cost - current_cost
        
        # Acceptance
        if delta_cost < 0 or (T > 0 and random.random() < pow(2.718281828, -delta_cost / T)):
            # Accept
            current_cost = new_cost
            current_conflicts = new_conflicts
            current_max = new_global_max
            
            if current_conflicts == 0 and current_max < best_max:
                best_max = current_max
                best_rows = copy.deepcopy(current_rows)
                best_conflicts = 0
        else:
            # Revert
            current_rows[r][j] = v
            
            # Revert diff_counts
            for d in new_diffs:
                c = diff_counts[d]
                diff_counts[d] = c - 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
                    
            for d in old_diffs:
                c = diff_counts.get(d, 0)
                diff_counts[d] = c + 1
                
            # Revert state vars
            # We don't need to recompute conflicts/max for rejected move if we revert correctly,
            # but to be safe against bugs, we can rely on the fact that we reverted the state.
            # However, current_conflicts and current_max were NOT updated in the reject branch.
            # So they are still correct for the previous state.
            # Wait, in the accept branch we updated them. In reject, we didn't.
            # So current_conflicts and current_max are still valid for the state BEFORE the move.
            # And we reverted the row and diff_counts.
            # So state is consistent.

        T *= cooling

    return best_rows
