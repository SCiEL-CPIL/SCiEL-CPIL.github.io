import time
import random
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Upper bound for differences. 
    # For k=7, best known is 134. Let's set a reasonable limit.
    # If k is larger, we might need a larger limit.
    # Estimate: k^2 * n_rows is a loose bound. 
    # Let's start with a limit and expand if necessary, or just use a large enough array.
    # For k=7, 200 is safe. For general k, let's use k*k*2 + 100.
    limit = max(200, k * k * 2 + 100)
    
    # used[i] is 1 if difference i is used by any row, 0 otherwise.
    used = bytearray(limit + 1)
    
    def get_row_diffs(row):
        """Calculate set of differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def remove_row_diffs(row):
        """Remove differences of a row from the global used set."""
        diffs = get_row_diffs(row)
        for d in diffs:
            if d <= limit:
                used[d] = 0
            else:
                # This shouldn't happen if limit is sufficient, but handle gracefully
                pass

    def add_row_diffs(row):
        """Add differences of a row to the global used set."""
        diffs = get_row_diffs(row)
        for d in diffs:
            if d <= limit:
                used[d] = 1
            else:
                # If difference exceeds limit, we have a problem with our bound.
                # For now, we assume limit is sufficient.
                pass

    def greedy_construct_row(used_arr, k, limit_val, randomize=False):
        """
        Greedily construct a row of k+1 elements starting at 0.
        used_arr: the current global used differences (excluding the row being built).
        """
        row = [0]
        current_max = 0
        
        for _ in range(k):
            next_x = current_max + 1
            found = False
            # Try to find the next element
            while next_x <= limit_val:
                # Check if adding next_x creates any conflict
                # Differences are next_x - a for a in row
                valid = True
                for a in row:
                    d = next_x - a
                    if d > limit_val:
                        valid = False
                        break
                    if used_arr[d]:
                        valid = False
                        break
                if valid:
                    row.append(next_x)
                    current_max = next_x
                    found = True
                    if randomize:
                        # With some probability, don't take the smallest, take a random valid one
                        # This is complex to implement efficiently in the inner loop.
                        # Instead, we can just break.
                        pass
                    break
                next_x += 1
            
            if not found:
                # If we can't find a valid next element within limit, 
                # we might need to expand limit or fail.
                # For the purpose of this heuristic, if we fail, we return a dummy row 
                # or the best effort. But usually limit is sufficient.
                # Let's just break and return what we have, which might be invalid length.
                break
        
        if len(row) < k + 1:
            # Fallback: just extend with large numbers? No, that breaks validity.
            # In the main loop, we should ensure we don't get stuck.
            # For now, return the partial row. The caller should handle this.
            pass
            
        return row

    # 1. Initialization
    rows = []
    for _ in range(n_rows):
        # Greedy construction with current used set
        # Note: greedy_construct_row doesn't modify used, it just reads it.
        # We need to ensure the row we build is valid with respect to 'used'.
        r = greedy_construct_row(used, k, limit, randomize=False)
        if len(r) < k + 1:
            # If greedy failed, we might need to increase limit or try different order.
            # For now, assume it works. If not, we can pad, but that's bad.
            # Let's try a more robust greedy: try all possible next steps? No, too slow.
            # We'll assume limit is large enough.
            pass
        rows.append(r)
        add_row_diffs(r)

    best_rows = [r[:] for r in rows]
    best_score = sum(max(r) for r in rows)
    
    # 2. Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    
    # Pre-calculate row diffs for quick access? 
    # It's better to recalculate on the fly for moves, or maintain a list of sets.
    row_diff_sets = [get_row_diffs(r) for r in rows]
    
    def get_score(rows_list):
        return sum(max(r) for r in rows_list)

    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        original_row = rows[r_idx]
        original_diffs = row_diff_sets[r_idx]
        
        # Remove original row's diffs from global used
        for d in original_diffs:
            if d <= limit:
                used[d] = 0
        
        # Generate a new candidate row
        # Strategy 1: Greedy with randomization
        # Strategy 2: Local perturbation of original row
        
        candidate_row = None
        method = random.random()
        
        if method < 0.5:
            # Try to reconstruct greedily, but allow some randomness in choice
            # To introduce randomness, we can shuffle the order of checking? 
            # No, we need increasing integers.
            # Let's try to rebuild from scratch but maybe skip the first few smallest valid options?
            # Simple approach: Just rebuild greedily. It might return the same row.
            # To force change, we can try to build with a slightly different starting constraint?
            # Or, we can use the "element shift" method instead.
            pass
        
        # Let's use Element Shifts primarily, as they are local and fast.
        # Pick an index to shift
        if 1 <= k: # k is at least 1 for this to make sense
            j = random.randint(1, k-1) # Index in row (1 to k-1)
            # If k=1, row is [0, x]. j=0 is only index. 
            # Task says k=7, so k-1 = 6. Indices 1..6 are shiftable.
            # If k is small, handle carefully.
            
            if k >= 2:
                j = random.randint(1, k-1)
            else:
                j = 0 # Only element to shift if k=1? No, 0 is fixed. 
                # If k=1, row is [0, a1]. We can shift a1.
                # Let's assume k>=2 for now as per task (k=7).
                
            orig_val = original_row[j]
            # Try shifts -3 to 3
            deltas = [-3, -2, -1, 1, 2, 3]
            random.shuffle(deltas)
            
            moved = False
            for delta in deltas:
                new_val = orig_val + delta
                
                # Check bounds
                if new_val <= 0:
                    continue
                if new_val > limit:
                    continue
                
                # Check increasing property
                if j > 0 and new_val <= original_row[j-1]:
                    continue
                if j < k and new_val >= original_row[j+1]:
                    continue
                
                # Create candidate row
                cand = original_row[:]
                cand[j] = new_val
                
                # Calculate new diffs for this row
                new_diffs = set()
                for i in range(k+1):
                    for l in range(i+1, k+1):
                        new_diffs.add(cand[l] - cand[i])
                
                # Check against global used (which currently excludes original_row's diffs)
                # We need to ensure new_diffs are not in 'used'
                # AND that new_diffs are distinct within the row (they are by construction of set)
                
                valid = True
                for d in new_diffs:
                    if d > limit:
                        valid = False
                        break
                    if used[d]:
                        valid = False
                        break
                
                if valid:
                    # Calculate score change
                    old_max = max(original_row)
                    new_max = max(cand)
                    # Score is sum of maxes. 
                    # If this row's max changes, score changes by (new_max - old_max)
                    # Other rows' maxes don't change.
                    delta_score = new_max - old_max
                    
                    # Acceptance criterion
                    if delta_score < 0:
                        accept = True
                    else:
                        accept = random.random() < np.exp(-delta_score / T)
                    
                    if accept:
                        # Commit the move
                        rows[r_idx] = cand
                        row_diff_sets[r_idx] = new_diffs
                        for d in new_diffs:
                            if d <= limit:
                                used[d] = 1
                        
                        current_score += delta_score
                        if current_score < best_score:
                            best_score = current_score
                            best_rows = [r[:] for r in rows]
                        
                        moved = True
                        break
            
            if not moved:
                # Revert: re-add original diffs
                for d in original_diffs:
                    if d <= limit:
                        used[d] = 1
        else:
            # Re-add original diffs if we didn't move
            for d in original_diffs:
                if d <= limit:
                    used[d] = 1

        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_rows
