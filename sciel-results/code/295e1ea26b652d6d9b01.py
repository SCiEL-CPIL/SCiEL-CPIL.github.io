"""Improved program for difference triangle set (4,7) using CSP with bitsets and backtracking."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct over all rows.
    Minimizes the largest entry.
    
    Uses CSP formulation with bitset-based difference tracking and strong forward checking.
    Each row is constructed sequentially by selecting marks that don't conflict with
    the global difference set.
    """
    # Total number of differences per row: C(k+1, 2) = k*(k+1)/2
    # For k=7: 28 differences per row, 4 rows: 112 total differences
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    # Counting bound: max_entry must be at least total_diffs_needed
    counting_bound = total_diffs_needed
    
    # Best known upper bound for (4,7) is 134
    # We'll try to find solutions starting from the counting bound upward
    
    def try_construct(max_entry, timeout=110):
        """Try to construct n_rows rows with max entry <= max_entry using backtracking with bitsets."""
        start_time = time.time()
        
        # Bitset representation of used differences
        # Use Python's int as a dynamic bitset for efficiency
        # Bit i is set if difference i is used
        used_diffs = 0  # bit i set means diff i is used
        
        rows = []
        
        def build_row(row_marks, current_max, used_diffs_local, depth=0):
            """
            Recursively build a single row using backtracking with strong forward checking.
            row_marks: list of marks in current row (starts with [0])
            current_max: current maximum mark
            used_diffs_local: bitset of used differences
            Returns: (success, row) or (False, None)
            """
            if len(row_marks) == k + 1:
                return True, row_marks
            
            # Check timeout
            if time.time() - start_time > timeout:
                return False, None
            
            # Number of remaining marks to place
            remaining_marks = k + 1 - len(row_marks)
            
            # Forward checking: we need 'remaining_marks' more marks in [current_max+1, max_entry]
            # The smallest possible next mark is current_max + 1
            # We need enough space: max_entry - current_max >= remaining_marks
            if max_entry - current_max < remaining_marks:
                return False, None
            
            # Try candidate next marks
            # Use heuristic: try marks that create "popular" differences first, or spread out
            candidates = list(range(current_max + 1, max_entry + 1))
            
            # Shuffle candidates for randomization to avoid getting stuck in local minima
            random.shuffle(candidates)
            
            for next_mark in candidates:
                # Check all differences between next_mark and existing marks
                valid = True
                new_diff_bits = 0
                
                for existing in row_marks:
                    diff = next_mark - existing
                    # Check if diff is already used
                    if (used_diffs_local >> diff) & 1:
                        valid = False
                        break
                    new_diff_bits |= (1 << diff)
                
                if not valid:
                    continue
                
                # Check that new differences don't conflict with each other
                # Since row_marks are strictly increasing and next_mark > all existing,
                # all diffs are positive and distinct by construction
                
                # Forward check: after placing this mark, can we still complete the row?
                # Remaining marks after this: remaining_marks - 1
                # We need remaining_marks - 1 more marks in [next_mark+1, max_entry]
                if remaining_marks - 1 > 0:
                    if max_entry - next_mark < remaining_marks - 1:
                        continue
                
                # Additional forward checking: count available differences
                # After placing this mark, we'll have used_diffs_local | new_diff_bits
                # The remaining differences to place are (remaining_marks - 1) marks,
                # each adding (len(row_marks) + 1) new differences
                # Total new diffs needed: sum_{i=1}^{remaining_marks-1} (len(row_marks) + i)
                # = (remaining_marks - 1) * len(row_marks) + (remaining_marks - 1) * remaining_marks // 2
                
                # This is a weak check; the main pruning comes from the bitset check
                
                # Update used_diffs
                updated_diffs = used_diffs_local | new_diff_bits
                
                # Recurse
                success, result_row = build_row(
                    row_marks + [next_mark], next_mark, updated_diffs, depth + 1
                )
                
                if success:
                    return True, result_row
            
            return False, None
        
        # Build rows one by one
        for row_idx in range(n_rows):
            if time.time() - start_time > timeout:
                return None
            
            success, row = build_row([0], 0, used_diffs)
            if not success:
                return None
            
            # Update global used_diffs with all differences in this row
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    used_diffs |= (1 << diff)
            
            rows.append(row)
        
        return rows
    
    # Try to find a solution with max entry as small as possible
    # Start from the counting bound, go up to 134 (best known)
    # For efficiency, we can try a few random restarts at each max_entry
    
    max_attempts_per_entry = 5
    max_entry_start = counting_bound
    max_entry_end = 140  # Slightly above best known
    
    for max_entry in range(max_entry_start, max_entry_end + 1):
        for attempt in range(max_attempts_per_entry):
            # Use different random seeds for each attempt
            random.seed(attempt * 1000 + max_entry)
            result = try_construct(max_entry, timeout=20)
            if result is not None:
                return result
    
    # If we didn't find a solution, fall back to a greedy approach
    # that might give a larger but valid solution
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
