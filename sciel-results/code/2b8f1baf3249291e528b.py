"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def _greedy_fallback(n_rows, k):
    """Greedy construction as fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _scope(rows):
    return max(max(row) for row in rows)

def _validate(rows):
    """Check if all differences across all rows are distinct."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _randomized_greedy(n_rows, k, rng, window=5):
    """
    Randomized greedy construction with lookahead.
    """
    used, rows = set(), []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            candidates = []
            # Look ahead a window of candidates
            for cand in range(x + 1, x + 1 + window * 2):
                new = {cand - a for a in row}
                if not new & used:
                    # Score: prefer candidates that introduce smaller differences
                    # to keep the row compact
                    score = sum(new)
                    candidates.append((score, cand))
            
            if not candidates:
                # If no candidates in window, extend search
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                else:
                    # Should not happen if we keep increasing x
                    continue
            else:
                # Sort by score (smallest sum of differences)
                candidates.sort()
                # Pick with some randomness to explore different structures
                if rng.random() < 0.6:
                    chosen = candidates[0][1]
                else:
                    # Pick randomly from top 3
                    top_n = min(3, len(candidates))
                    chosen = candidates[rng.randrange(top_n)][1]
                
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
        
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, time_limit, rng):
    """
    Local search to reduce scope.
    """
    start = time.time()
    best = [row[:] for row in rows]
    best_scope = _scope(best)
    
    # Precompute differences for each row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Global used differences
    all_diffs = set()
    row_diffs = [get_row_diffs(row) for row in best]
    for dset in row_diffs:
        all_diffs |= dset

    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Shuffle order of rows and positions to avoid bias
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for r in row_order:
            row = best[r]
            # Try to reduce each entry from right to left
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try to decrease to the smallest valid value > row[pos-1]
                lower_bound = row[pos - 1] + 1
                if old_val <= lower_bound:
                    continue
                
                # Try values from lower_bound up to old_val - 1
                # We want the smallest valid value to minimize scope
                for new_val in range(lower_bound, old_val):
                    # Check if new_val creates any conflict
                    # 1. Check within the row
                    # 2. Check against other rows
                    
                    # Temporarily update row
                    row[pos] = new_val
                    
                    # Compute new diffs for this row
                    new_row_diffs = get_row_diffs(row)
                    
                    # Check internal consistency (should be fine if we just computed it, 
                    # but we need to ensure no duplicates within the row)
                    # get_row_diffs uses a set, so if len < expected, there was a dup
                    expected_diffs = len(row) * (len(row) - 1) // 2
                    if len(new_row_diffs) != expected_diffs:
                        row[pos] = old_val
                        continue
                    
                    # Check against other rows
                    # The differences that were in row_diffs[r] are no longer valid
                    # The differences in new_row_diffs must not intersect with 
                    # (all_diffs - row_diffs[r])
                    
                    other_diffs = all_diffs - row_diffs[r]
                    
                    if new_row_diffs & other_diffs:
                        row[pos] = old_val
                        continue
                    
                    # Valid move!
                    # Update global state
                    all_diffs -= row_diffs[r]
                    all_diffs |= new_row_diffs
                    row_diffs[r] = new_row_diffs
                    
                    # Update scope
                    current_scope = _scope(best)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        improved = True
                        # If we reduced the global max, we can break early for this row
                        # but we continue to try to reduce other entries in this row
                        # or other rows
                    break # Found the smallest valid value, stop searching higher for this pos

    return best, best_scope

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # 1. Get a greedy solution as fallback
    greedy_rows = _greedy_fallback(n_rows, k)
    best_rows = greedy_rows
    best_scope = _scope(greedy_rows)
    
    # 2. Run randomized greedy with restarts
    rng = random.Random(42)
    
    # Determine how many restarts we can afford
    # Each restart + local search should take ~1-5 seconds
    # Let's try to do at least 10 restarts if time permits
    
    while time.time() - start_time < deadline - 10:
        # Generate a candidate
        candidate_rows = _randomized_greedy(n_rows, k, rng, window=5)
        cand_scope = _scope(candidate_rows)
        
        if cand_scope < best_scope:
            best_rows = candidate_rows
            best_scope = cand_scope
            
            # If we found a solution close to the lower bound, we might be done
            # Lower bound is roughly n_rows * k*(k+1)/2
            num_diffs = n_rows * k * (k + 1) // 2
            if best_scope <= num_diffs + 10:
                break
        
        # 3. Local search to improve
        remaining_time = deadline - time.time() - 5
        if remaining_time > 2:
            # Allocate a fraction of remaining time to local search
            # Don't spend too much time on one candidate if it's not promising
            search_time = min(remaining_time * 0.3, 5.0)
            
            improved_rows, improved_scope = _local_search(
                best_rows, n_rows, k, search_time, rng
            )
            
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows
                
                # If we hit the lower bound, we can stop
                num_diffs = n_rows * k * (k + 1) // 2
                if best_scope <= num_diffs + 5:
                    break
    
    # 4. Final local search on the best solution if time permits
    remaining_time = deadline - time.time() - 2
    if remaining_time > 3:
        improved_rows, improved_scope = _local_search(
            best_rows, n_rows, k, remaining_time, rng
        )
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
            
    return best_rows
# EVOLVE-BLOCK-END
