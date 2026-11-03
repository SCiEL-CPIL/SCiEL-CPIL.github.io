import time
import random
from itertools import combinations

def construct(n_rows, k):
    # Deadline guard: return best valid result before 115s
    deadline = time.time() + 115
    
    # Total number of differences needed per row: k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on M: need at least total_diffs distinct positive integers in [1, M-1]
    # So M-1 >= total_diffs => M >= total_diffs + 1
    # But also each row's largest entry >= max difference in that row
    # Simple counting bound: M >= total_diffs + 1
    M_lower = total_diffs + 1
    
    # Try to find a good solution using a combination of:
    # 1. Greedy construction (baseline)
    # 2. Local search / hill climbing on the set of differences
    
    def greedy_construct(n_rows, k):
        """Greedy: each row takes smallest next mark with no difference conflicts."""
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
    
    def is_valid(rows, n_rows, k):
        """Check if all positive differences within rows are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        # Verify structure
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    def scope_of(rows):
        """Largest entry across all rows."""
        return max(max(row) for row in rows)
    
    # Start with greedy
    best_rows = greedy_construct(n_rows, k)
    best_scope = scope_of(best_rows)
    
    if not is_valid(best_rows, n_rows, k):
        # Shouldn't happen with greedy, but safety
        return best_rows
    
    # Now try to improve via local search on the rows
    # Representation: rows as list of lists
    # Moves:
    # 1. For a given row, try to replace one element with a smaller valid one
    # 2. Try to reduce the largest element in a row
    
    def try_reduce_row(row, used_other, n_rows, k):
        """
        Try to find a smaller valid row given that used_other is the set of differences
        used by all other rows. Returns a new row if found, else None.
        """
        # We want to minimize the max element
        # Try greedy construction but with a cap
        current_max = max(row)
        
        # Try to build a row with max element < current_max
        for max_val in range(current_max - 1, 0, -1):
            # Try to find k+1 elements starting at 0, ending <= max_val
            # with all pairwise differences not in used_other and distinct within row
            row_new = [0]
            used_in_row = set()
            x = 0
            success = True
            while len(row_new) < k + 1:
                x += 1
                if x > max_val:
                    success = False
                    break
                new_diffs = {x - a for a in row_new}
                if new_diffs & used_other or new_diffs & used_in_row:
                    continue
                row_new.append(x)
                used_in_row |= new_diffs
            if success:
                return row_new
        return None
    
    # Local search: iteratively try to reduce each row
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r_idx in range(n_rows):
            if time.time() > deadline - 5:
                break
            # Compute used differences from all other rows
            used_other = set()
            for r2_idx in range(n_rows):
                if r2_idx == r_idx:
                    continue
                row2 = best_rows[r2_idx]
                for i in range(len(row2)):
                    for j in range(i + 1, len(row2)):
                        used_other.add(row2[j] - row2[i])
            
            new_row = try_reduce_row(best_rows[r_idx], used_other, n_rows, k)
            if new_row is not None and max(new_row) < max(best_rows[r_idx]):
                best_rows[r_idx] = new_row
                best_scope = scope_of(best_rows)
                improved = True
    
    # Second phase: try randomized restarts with different orderings
    if time.time() < deadline - 10:
        # Try different row orderings
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            order = list(range(n_rows))
            random.shuffle(order)
            
            used = set()
            rows = [None] * n_rows
            success = True
            for r_idx in order:
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        continue
                    row.append(x)
                    used |= new_diffs
                rows[r_idx] = row
            
            if is_valid(rows, n_rows, k):
                s = scope_of(rows)
                if s < best_scope:
                    best_rows = rows
                    best_scope = s
    
    # Third phase: try to further optimize by swapping elements
    if time.time() < deadline - 5:
        # For each row, try to replace an element with a smaller one
        for r_idx in range(n_rows):
            if time.time() > deadline - 2:
                break
            row = best_rows[r_idx]
            used_other = set()
            for r2_idx in range(n_rows):
                if r2_idx == r_idx:
                    continue
                row2 = best_rows[r2_idx]
                for i in range(len(row2)):
                    for j in range(i + 1, len(row2)):
                        used_other.add(row2[j] - row2[i])
            
            # Try to reduce each position from the end
            for pos in range(k, 0, -1):
                if time.time() > deadline - 2:
                    break
                current_val = row[pos]
                # Try values from current_val-1 down to row[pos-1]+1
                for new_val in range(current_val - 1, row[pos-1], -1):
                    # Check if replacing row[pos] with new_val keeps all diffs valid
                    new_row = row[:pos] + [new_val] + row[pos+1:]
                    # Check diffs within new_row
                    row_diffs = set()
                    valid = True
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in row_diffs or d in used_other:
                                valid = False
                                break
                            row_diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        best_rows[r_idx] = new_row
                        best_scope = scope_of(best_rows)
                        break  # Move to next position
    
    return best_rows
