"""Difference triangle set (4,7): construct rows with distinct global differences."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 115.0  # safety margin before 120s kill
    total_differences = n_rows * (k + 1) * k // 2
    
    # Lower bound on scope: we need total_differences distinct values in [1, scope]
    # So scope >= total_differences
    lb = total_differences
    
    # We'll try to find a solution with scope close to lb.
    # Strategy: randomized greedy with backtracking, multiple restarts.
    
    best_rows = None
    best_scope = float('inf')
    
    # First, get a quick greedy solution to have a fallback
    def greedy_construct():
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
    
    initial_rows = greedy_construct()
    best_rows = initial_rows
    best_scope = max(max(r) for r in best_rows)
    
    # Now try randomized approaches to improve
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Randomize the order of rows and use randomized greedy
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        used = set()
        rows = [None] * n_rows
        success = True
        
        for row_idx in row_order:
            row = [0]
            # Try to build this row with some randomness
            attempts = 0
            while len(row) < k + 1 and attempts < 5000:
                attempts += 1
                if time.time() > deadline:
                    success = False
                    break
                
                # Choose next value: try a random range of candidates
                # Bias towards smaller values but allow some exploration
                if len(row) == 1:
                    # First step after 0
                    start_candidate = 1
                    end_candidate = min(best_scope, 20)
                else:
                    last = row[-1]
                    start_candidate = last + 1
                    end_candidate = min(best_scope, last + 20)
                
                # Try candidates in a randomized order, but prefer smaller
                candidates = list(range(start_candidate, end_candidate + 1))
                random.shuffle(candidates)
                # Sort with a bias: keep small ones more likely at front
                candidates.sort(key=lambda x: random.random() ** 0.5 * x)
                
                placed = False
                for cand in candidates:
                    if cand > best_scope:
                        continue
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        row.append(cand)
                        used |= new_diffs
                        placed = True
                        break
                
                if not placed:
                    # Try a wider range
                    wide_start = start_candidate
                    wide_end = best_scope
                    if wide_end > wide_start:
                        wide_candidates = list(range(wide_start, wide_end + 1))
                        random.shuffle(wide_candidates)
                        for cand in wide_candidates:
                            new_diffs = {cand - a for a in row}
                            if not new_diffs & used:
                                row.append(cand)
                                used |= new_diffs
                                placed = True
                                break
                
                if not placed:
                    success = False
                    break
            
            if not success:
                break
            
            rows[row_idx] = row
        
        if success and all(rows[i] is not None for i in range(n_rows)):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                # Update lower bound check
                if best_scope == lb:
                    break
    
    # Final fallback: ensure we return a valid solution
    if best_rows is None:
        best_rows = initial_rows
    
    return best_rows
# EVOLVE-BLOCK-END
