"""Difference triangle set (7,5) - find 7 rows of 7 marks minimizing max entry."""
import time
import random
import copy

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_row(used, k):
    """Build one row greedily: start at 0, pick smallest next mark with no diff conflicts."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
    return rows

def _generate_sidon_rows(k, max_val):
    """
    Generate all valid rows (Sidon sets) of length k+1 starting with 0,
    strictly increasing, with max element <= max_val.
    Returns a list of tuples, each tuple containing the row and its difference set.
    """
    valid_rows = []
    
    # Recursive generation
    def backtrack(current_row, current_diffs):
        if len(current_row) == k + 1:
            # Valid row found
            valid_rows.append((tuple(current_row), frozenset(current_diffs)))
            return
        
        last = current_row[-1]
        # Next element must be > last and <= max_val
        # Also, we need to ensure we can still fit the remaining elements.
        # Remaining elements needed: (k + 1) - len(current_row)
        # The smallest possible next value is last + 1.
        # The largest possible value is max_val.
        
        for next_val in range(last + 1, max_val + 1):
            new_diffs = {next_val - a for a in current_row}
            if new_diffs & current_diffs:
                continue
            
            # Check if we can still complete the row
            # We need (k + 1 - len(current_row) - 1) more elements after next_val.
            # The minimum span required for the remaining elements is roughly 
            # (remaining_count) * (remaining_count + 1) / 2 ? No, just distinct diffs.
            # Simple bound: next_val + remaining_count <= max_val is necessary but not sufficient.
            remaining = k + 1 - len(current_row) - 1
            if next_val + remaining > max_val:
                break
            
            current_row.append(next_val)
            current_diffs = current_diffs | new_diffs
            backtrack(current_row, current_diffs)
            current_row.pop()
            current_diffs = current_diffs - new_diffs # This is inefficient, better to pass a set and modify/copy or use list of diffs
            
    # Optimization: Use a list for diffs to make add/remove O(1) or just pass a set and copy.
    # Given the small depth (6), copying sets is fine.
    
    # Let's re-implement backtrack with proper state management
    valid_rows.clear()
    
    def backtrack2(row, diffs):
        if len(row) == k + 1:
            valid_rows.append((tuple(row), frozenset(diffs)))
            return
        
        last = row[-1]
        remaining = k + 1 - len(row)
        
        # Pruning: max possible value for last element is max_val
        # Min possible value for next is last + 1
        # We need 'remaining' elements.
        # The smallest sequence is last+1, last+2, ..., last+remaining
        # So we need last + remaining <= max_val
        
        for next_val in range(last + 1, max_val + 1):
            if next_val + remaining - 1 > max_val:
                break
            
            new_diffs = set()
            valid = True
            for a in row:
                d = next_val - a
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            row.append(next_val)
            diffs |= new_diffs
            backtrack2(row, diffs)
            diffs -= new_diffs
            row.pop()

    backtrack2([0], set())
    return valid_rows

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # Fallback
    best_rows = _greedy_all(n_rows, k)
    best_score = _max_entry(best_rows)
    
    # Theoretical lower bound for max entry M
    # Total differences = n_rows * (k+1)*k/2
    # Max difference <= M
    # So M >= n_rows * (k+1)*k/2
    total_diffs = n_rows * (k + 1) * k // 2
    min_M = total_diffs
    
    # Generate rows up to a reasonable upper bound
    # Best known for (7,5) is 112. Let's generate up to 115 to be safe.
    # If we find a solution at M, we return it.
    
    # We need to generate rows once for the max M we might check.
    # But generating rows up to 115 might be slow if done naively.
    # Let's try to generate rows incrementally or just for the specific M we check.
    # Actually, generating all rows up to 115 is a one-time cost.
    
    max_search_M = 115
    all_rows = _generate_sidon_rows(k, max_search_M)
    
    if not all_rows:
        return best_rows
    
    # Sort rows by max element
    all_rows.sort(key=lambda x: x[0][-1])
    
    # Iterate M from min_M to max_search_M
    for M in range(min_M, max_search_M + 1):
        if time.time() > deadline - 10:
            break
            
        # Filter rows with max element <= M
        # Since all_rows is sorted, we can find the index
        # But we need to ensure we only use rows with max <= M.
        
        # Optimization: Pre-filter for each M? 
        # Or just iterate through all_rows and pick those with max <= M.
        
        candidates = []
        for row, diffs in all_rows:
            if row[-1] <= M:
                candidates.append((row, diffs))
            else:
                break # Since sorted
        
        if len(candidates) < n_rows:
            continue
        
        # Try to find n_rows compatible rows from candidates
        # Backtracking search
        found = False
        selected = []
        
        def backtrack_select(idx, selected_rows, selected_diffs):
            nonlocal found
            if len(selected_rows) == n_rows:
                found = True
                return True
            
            # Pruning: if not enough candidates left, return
            if len(candidates) - idx < n_rows - len(selected_rows):
                return False
            
            # Heuristic: try candidates that are "compatible" with current selection
            # To improve speed, we can sort candidates by max element (already done)
            # and iterate.
            
            for i in range(idx, len(candidates)):
                row, diffs = candidates[i]
                
                # Check compatibility with all selected rows
                compatible = True
                for sel_diffs in selected_diffs:
                    if diffs & sel_diffs:
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # Select this row
                selected_rows.append(row)
                selected_diffs.append(diffs)
                
                if backtrack_select(i + 1, selected_rows, selected_diffs):
                    return True
                
                selected_rows.pop()
                selected_diffs.pop()
                
                # Check time
                if time.time() > deadline - 5:
                    return False
            
            return False
        
        if backtrack_select(0, [], []):
            # Solution found
            result_rows = [list(r) for r in selected_rows]
            score = _max_entry(result_rows)
            if score < best_score:
                best_rows = result_rows
                best_score = score
            return best_rows
    
    return best_rows

