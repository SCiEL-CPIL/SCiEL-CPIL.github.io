"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the maximum entry (scope), subject to all positive differences
    within each row being distinct across all rows.
    
    Strategy:
    1. Build initial rows using a 'largest-first' greedy approach with conflict tracking.
    2. Use conflict-directed repair to reduce the scope by replacing large entries
       with smaller ones that don't create conflicts.
    3. Deadline guard to ensure we finish within 120s.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Total number of differences needed: n_rows * k*(k+1)/2
    # Each difference d must be unique across all rows.
    
    # Step 1: Build initial rows using greedy largest-first placement
    # We want to minimize the max entry, so we try to keep entries small.
    # Use a greedy approach that picks the smallest possible next value
    # that doesn't conflict with existing differences.
    
    used_diffs = set()
    rows = []
    
    for r in range(n_rows):
        row = [0]
        # Greedy: try to build the row with smallest possible values
        # But we need k more elements after 0
        for pos in range(k):
            # Try smallest possible x > row[-1]
            x = row[-1] + 1
            while True:
                # Check if adding x creates any new difference that's already used
                new_diffs = set()
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        new_diffs = None
                        break
                    new_diffs.add(d)
                if new_diffs is not None:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    break
                x += 1
        rows.append(row)
    
    # Get current scope
    current_scope = max(max(row) for row in rows)
    
    # Step 2: Conflict-directed repair to reduce scope
    # Try to reduce the largest entries by replacing them with smaller values
    # that don't create conflicts
    
    best_rows = [row[:] for row in rows]
    best_scope = current_scope
    
    # Rebuild used_diffs for best_rows
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    used_diffs = compute_used_diffs(best_rows)
    
    # Try to reduce scope by iterating over rows and trying to shrink large entries
    # Focus on the largest entries first
    for iteration in range(50):  # limited iterations
        if time.time() > deadline:
            break
        
        # Find the row and position with the largest entry
        max_val = 0
        max_row_idx = -1
        max_pos = -1
        for r_idx, row in enumerate(best_rows):
            for p_idx in range(1, len(row)):
                if row[p_idx] > max_val:
                    max_val = row[p_idx]
                    max_row_idx = r_idx
                    max_pos = p_idx
        
        if max_row_idx == -1:
            break
        
        # Try to reduce this entry
        row = best_rows[max_row_idx]
        old_val = row[max_pos]
        
        # Try values from max_val - 1 down to row[max_pos-1] + 1
        found = False
        for new_val in range(old_val - 1, row[max_pos-1], -1):
            if time.time() > deadline:
                break
            
            # Check if replacing old_val with new_val maintains validity
            # Remove old differences involving old_val
            old_diffs = set()
            for a in row:
                if a != old_val:
                    d = abs(old_val - a)
                    old_diffs.add(d)
            
            # Add new differences involving new_val
            new_diffs = set()
            for a in row:
                if a != new_val:
                    d = abs(new_val - a)
                    new_diffs.add(d)
            
            # Check for conflicts: new_diffs should not intersect with (used_diffs - old_diffs)
            # Also, new_diffs should be internally consistent (no duplicates within new_diffs)
            remaining_used = used_diffs - old_diffs
            
            # Check if any new difference conflicts with remaining used differences
            conflict = False
            for d in new_diffs:
                if d in remaining_used:
                    conflict = True
                    break
            
            # Also check that new_diffs doesn't have internal conflicts with each other
            # (they're all from the same row, so they just need to not conflict with other rows)
            
            if not conflict:
                # Also ensure the row is still strictly increasing
                if max_pos > 0 and new_val <= row[max_pos-1]:
                    continue
                if max_pos < len(row) - 1 and new_val >= row[max_pos+1]:
                    continue
                
                # Valid replacement
                row[max_pos] = new_val
                used_diffs = (used_diffs - old_diffs) | new_diffs
                current_scope = max(max(row2) for row2 in best_rows)
                
                if current_scope < best_scope:
                    best_scope = current_scope
                
                found = True
                break
        
        if not found:
            # Try other large entries
            # Find second largest, etc.
            # For simplicity, just break if we can't reduce the largest
            # Try a different row
            pass
    
    # Step 3: Local search - try to improve by swapping or adjusting entries
    # Try to reduce scope further by examining all entries in the row with max value
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = max(max(row) for row in best_rows)
        
        # Try to reduce each entry in rows that have the max value
        for r_idx in range(n_rows):
            if time.time() > deadline:
                break
            row = best_rows[r_idx]
            for p_idx in range(len(row) - 1, 0, -1):  # from largest to smallest
                if time.time() > deadline:
                    break
                
                old_val = row[p_idx]
                if old_val < best_scope:
                    continue  # Only try to reduce entries that are at or above current best scope
                
                # Try to reduce this entry
                for new_val in range(old_val - 1, row[p_idx-1], -1):
                    if time.time() > deadline:
                        break
                    
                    # Check validity
                    old_diffs = set()
                    for a in row:
                        if a != old_val:
                            old_diffs.add(abs(old_val - a))
                    
                    new_diffs = set()
                    for a in row:
                        if a != new_val:
                            new_diffs.add(abs(new_val - a))
                    
                    remaining_used = used_diffs - old_diffs
                    
                    conflict = False
                    for d in new_diffs:
                        if d in remaining_used:
                            conflict = True
                            break
                    
                    if not conflict:
                        if new_val <= row[p_idx-1] or (p_idx < len(row)-1 and new_val >= row[p_idx+1]):
                            continue
                        
                        row[p_idx] = new_val
                        used_diffs = (used_diffs - old_diffs) | new_diffs
                        new_scope = max(max(row2) for row2 in best_rows)
                        
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                        break
    
    return best_rows

# EVOLVE-BLOCK-END
