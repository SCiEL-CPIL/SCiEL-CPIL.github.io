"""Improved program for difference triangle set (4,7) using backtracking with bitsets."""
# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct over all rows.
    Minimizes the largest entry.
    
    Uses backtracking with bitset-based difference tracking and strong pruning.
    """
    # For k=7, n_rows=4: need 112 distinct differences
    # Best known max entry is 134
    
    # Strategy: Try to find solution with max_entry starting from a reasonable lower bound
    # Use randomized backtracking with good heuristics and time limit
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    counting_bound = total_diffs_needed  # 112 for n_rows=4, k=7
    
    # Try max_entry values from counting_bound upward
    # But start from a slightly higher value to avoid spending too much time on infeasible cases
    # We know 134 works, so try from around 120 to 140
    
    start_max = max(counting_bound, 118)
    end_max = 145
    
    for max_entry in range(start_max, end_max + 1):
        if max_entry < total_diffs_needed:
            continue
        
        result = _backtrack_search(n_rows, k, max_entry, time_limit=20)
        if result is not None:
            return result
    
    # Fallback: greedy construction (may give larger max entry)
    return _greedy_construct(n_rows, k)


def _backtrack_search(n_rows, k, max_entry, time_limit=20):
    """
    Backtracking search to find n_rows rows of k+1 marks starting at 0
    with all pairwise differences distinct across all rows.
    """
    start_time = time.time()
    
    # used_diffs[0..max_entry] tracks which differences are already used
    used_diffs = bytearray(max_entry + 1)
    rows = []
    
    # For each row, we build marks [0, m1, m2, ..., mk] where 0 < m1 < m2 < ... < mk <= max_entry
    # All differences mj - mi for i < j must be distinct and not in used_diffs
    
    def can_complete_row(current_marks, used_diffs_local):
        """Check if it's possible to complete the current row."""
        if len(current_marks) == k + 1:
            return True
        
        remaining_marks = k + 1 - len(current_marks)
        current_max = current_marks[-1] if current_marks else 0
        
        # We need remaining_marks more marks, each at least 1 apart
        # So the smallest possible last mark is current_max + remaining_marks
        if current_max + remaining_marks > max_entry:
            return False
        
        # Each new mark adds (len(current_marks)) new differences
        # Total remaining differences to assign: sum over next marks
        # For a new mark at position p (relative), it adds p differences
        # This is a weak check; we'll rely on the main backtracking
        
        return True
    
    def backtrack_row(row_marks, used_diffs_local):
        """
        Recursively build a single row.
        row_marks: list of marks in current row (starts with [0])
        used_diffs_local: bytearray tracking used differences
        Returns: list of marks if successful, None otherwise
        """
        if len(row_marks) == k + 1:
            return list(row_marks)
        
        current_max = row_marks[-1]
        remaining = k + 1 - len(row_marks)
        
        # Try next marks from current_max + 1 to max_entry
        # Use randomized order with bias toward smaller values for better packing
        candidates = list(range(current_max + 1, max_entry + 1))
        
        # Shuffle with bias: prefer smaller values but with some randomness
        random.shuffle(candidates)
        
        for next_mark in candidates:
            # Check if we can still fit remaining marks after this one
            if next_mark + (remaining - 1) > max_entry:
                continue
            
            # Check all differences between next_mark and existing marks
            valid = True
            new_diffs = []
            for existing in row_marks:
                diff = next_mark - existing
                if diff > max_entry or used_diffs_local[diff]:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            # Mark differences as used
            for d in new_diffs:
                used_diffs_local[d] = 1
            
            # Recurse
            row_marks.append(next_mark)
            result = backtrack_row(row_marks, used_diffs_local)
            
            if result is not None:
                return result
            
            # Undo
            row_marks.pop()
            for d in new_diffs:
                used_diffs_local[d] = 0
            
            # Time check
            if time.time() - start_time > time_limit:
                return None
        
        return None
    
    # Build rows one by one
    for row_idx in range(n_rows):
        if time.time() - start_time > time_limit:
            return None
        
        row = backtrack_row([0], used_diffs)
        if row is None:
            return None
        rows.append(row)
    
    return rows


def _greedy_construct(n_rows, k):
    """
    Greedy fallback: construct rows by always picking the smallest valid next mark.
    May give a larger max entry but guarantees a valid solution.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
