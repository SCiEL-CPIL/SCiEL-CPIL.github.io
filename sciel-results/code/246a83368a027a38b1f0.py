"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    num_diffs_per_row = k * (k + 1) // 2
    
    def greedy_row(used_diffs, start_bias=0):
        """Build a single row greedily, avoiding used differences."""
        row = [0]
        used_local = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs) and not (new_diffs & used_local):
                row.append(x)
                used_local |= new_diffs
        return row, used_local
    
    def row_diffs(row):
        """Get the set of positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    # Strategy: Try multiple random restarts with different orderings and biases
    best_rows = None
    best_max = float('inf')
    
    # First, try the simple greedy as a baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row, row_diffs_set = greedy_row(used)
        rows.append(row)
        used |= row_diffs_set
    if max_entry(rows) < best_max:
        best_max = max_entry(rows)
        best_rows = [r[:] for r in rows]
    
    # Now try randomized local search
    # We'll use a simulated annealing / hill climbing approach
    # where we try to rearrange or modify rows to reduce the max entry
    
    # Approach: Generate rows one by one, but with randomization in the greedy step
    # Then try to improve by swapping or modifying
    
    for attempt in range(200):
        random.seed(attempt)
        used = set()
        rows = []
        
        for i in range(n_rows):
            # Randomized greedy: sometimes skip a valid candidate to explore
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # With some probability, accept this; otherwise try next
                    if random.random() < 0.7:
                        row.append(x)
                        used |= new_diffs
                    # else: skip this x, try x+1
            rows.append(row)
        
        cur_max = max_entry(rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [r[:] for r in rows]
        
        # Local improvement: try to reduce the max entry by adjusting the last element
        # or by re-greedy-ing some rows
        improved = True
        while improved and best_max > 0:
            improved = False
            for idx in range(n_rows):
                # Try to rebuild row idx with a tighter bound
                # Remove its differences from used
                old_diffs = row_diffs(rows[idx])
                used_copy = used - old_diffs
                
                # Try to build a new row with smaller max
                new_row, new_diffs_set = greedy_row(used_copy)
                if max(new_row) < max(rows[idx]):
                    rows[idx] = new_row
                    used = used_copy | new_diffs_set
                    best_max = max_entry(rows)
                    improved = True
    
    # Final local search: try swapping elements between rows
    if best_rows is not None:
        for _ in range(50):
            rows = [r[:] for r in best_rows]
            used = set()
            for r in rows:
                used |= row_diffs(r)
            
            for i in range(n_rows):
                for j in range(i + 1, n_rows):
                    # Try swapping the last elements if it helps
                    if rows[i][-1] != rows[j][-1]:
                        a, b = rows[i][-1], rows[j][-1]
                        # Temporarily swap
                        rows[i][-1], rows[j][-1] = b, a
                        # Check validity
                        new_used = set()
                        valid = True
                        for r in rows:
                            diffs = row_diffs(r)
                            if diffs & new_used:
                                valid = False
                                break
                            new_used |= diffs
                        if valid:
                            new_max = max_entry(rows)
                            if new_max < best_max:
                                best_max = new_max
                                best_rows = [r[:] for r in rows]
                        else:
                            # Revert
                            rows[i][-1], rows[j][-1] = a, b
    
    return best_rows
# EVOLVE-BLOCK-END
