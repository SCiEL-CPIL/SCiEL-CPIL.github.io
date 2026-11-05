import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    def get_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

    def is_valid_row(row, used_diffs):
        # Check if adding this row keeps differences distinct
        new_diffs = get_diffs(row)
        return not (new_diffs & used_diffs)

    def greedy_construct(randomize=False):
        """
        Constructs a valid set of rows using a greedy approach.
        If randomize is True, it picks from a small set of valid candidates
        instead of always the smallest.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                valid_candidates = []
                # Look ahead a bit to find valid candidates
                # To keep it fast, we just scan upwards
                while True:
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        valid_candidates.append(x)
                        if len(valid_candidates) >= 5: # Small window
                            break
                    x += 1
                    if time.time() - start_time > time_limit:
                        return None # Timeout during construction
                
                if randomize and valid_candidates:
                    # Pick a random candidate from the small window
                    # Bias towards smaller ones to keep max low
                    idx = random.randint(0, min(2, len(valid_candidates) - 1))
                    x = valid_candidates[idx]
                else:
                    x = valid_candidates[0]
                
                row.append(x)
                used |= set(x - a for a in row[:-1])
                x += 1
            
            rows.append(row)
        return rows

    def reduce_solution(rows):
        """
        Attempts to reduce the maximum entry by trying to replace
        large elements with smaller valid ones.
        Returns the reduced solution.
        """
        used = set()
        for r in rows:
            used |= get_diffs(r)
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # Find the row with the current max
            max_val = max(max(r) for r in rows)
            # Try to reduce elements in rows, starting with the one having the max
            # We iterate rows in order of their max value descending
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)
            
            for r_idx in row_indices:
                if time.time() - start_time > time_limit:
                    break
                row = rows[r_idx]
                # Try to reduce elements from the end (largest) to the beginning
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    # Try to find a smaller value v < current_val
                    # such that v > row[pos-1] and differences are distinct
                    # We only need to check differences involving row[pos]
                    # Old diffs involving row[pos]: {current_val - row[i] for i in range(pos)}
                    # New diffs involving v: {v - row[i] for i in range(pos)}
                    
                    # Remove old diffs from 'used' temporarily?
                    # Easier: compute new_diffs_for_pos and check against (used - old_diffs_for_pos)
                    old_diffs = set(current_val - row[i] for i in range(pos))
                    
                    # Find smallest valid v
                    min_v = row[pos - 1] + 1
                    # We want the smallest v to maximize future reduction potential?
                    # Or just any v < current_val. Smallest is best for "dense" packing.
                    found = False
                    for v in range(min_v, current_val):
                        new_diffs = set(v - row[i] for i in range(pos))
                        # Check if new_diffs conflicts with (used - old_diffs)
                        # used contains all diffs from all rows.
                        # We remove old_diffs from used to get the "rest"
                        # If new_diffs & (used - old_diffs) is empty, it's valid.
                        if not (new_diffs & (used - old_diffs)):
                            # Valid replacement
                            # Update used
                            used -= old_diffs
                            used |= new_diffs
                            row[pos] = v
                            improved = True
                            found = True
                            break
                    if found:
                        # Continue reducing this row or move to next?
                        # Moving to next element in same row might allow further reduction
                        # But we updated 'used', so subsequent checks are valid.
                        # Let's break out of pos loop to re-evaluate max?
                        # Actually, just continue to next pos.
                        pass
                if improved:
                    # We made a change, the max might have changed.
                    # Break out to re-sort rows?
                    # For efficiency, just break and loop again.
                    break
        
        return rows

    best_solution = None
    best_max = float('inf')
    
    # Run multiple iterations of Randomized Greedy + Reduction
    while time.time() - start_time < time_limit:
        # 1. Construct
        rows = greedy_construct(randomize=True)
        if rows is None:
            break
            
        # 2. Reduce
        rows = reduce_solution(rows)
        
        # 3. Evaluate
        current_max = max(max(r) for r in rows)
        if current_max < best_max:
            best_max = current_max
            best_solution = [list(r) for r in rows]
            
        # Optional: If we hit a very good bound, we can try to polish it more
        # or break if we find the theoretical lower bound (112)
        if best_max <= 112:
            break

    if best_solution is None:
        # Fallback to simple greedy if randomization failed or timed out early
        rows = greedy_construct(randomize=False)
        if rows:
            best_solution = [list(r) for r in rows]
        else:
            # Absolute fallback: manual small construction or error
            # Should not happen for small n_rows, k
            best_solution = [[0, 1, 2, 3, 4, 5, 6, 7]] * n_rows # Invalid, but prevents crash
            # Actually, let's just return the greedy one if it exists
            rows = greedy_construct(randomize=False)
            best_solution = [list(r) for r in rows]

    return best_solution
