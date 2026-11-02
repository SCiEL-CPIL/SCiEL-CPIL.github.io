import random
import time
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set.
    n_rows: number of rows
    k: number of intervals, so k+1 points per row.
    Returns a list of n_rows rows, each a list of k+1 integers.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_solution = None
    best_score = float('inf')
    
    # Helper to check if a row is valid given a set of used differences
    def get_differences(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row, used_diffs):
        # Check if row is strictly increasing (assumed by construction)
        # Check if differences are distinct and not in used_diffs
        diffs = get_differences(row)
        if len(diffs) != len(row) * (len(row) - 1) // 2:
            return False
        if diffs & used_diffs:
            return False
        return True

    def greedy_construct_randomized(noise_level=0.0):
        """
        Constructs a set of rows using a greedy approach with randomization.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We want to find k more numbers
            # Candidate values: start from 1 upwards
            # To allow randomization, we can look at a window of candidates
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates
                valid_candidates = []
                # Limit search space for speed, usually the answer is < 200
                # But we don't know M yet. Let's search up to a reasonable bound.
                # If we can't find a candidate, we might need to restart or increase bound.
                # For initialization, a high bound is safe.
                upper_bound = 300 
                start_val = row[-1] + 1
                
                # Optimization: Instead of checking all, we can be smarter.
                # But for n=7, k=5, 300 is small.
                found = False
                for x in range(start_val, upper_bound + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        # Valid candidate
                        valid_candidates.append(x)
                        # If noise is low, we might take the first one.
                        # If noise is high, we collect a few.
                        if len(valid_candidates) >= 5:
                            break
                
                if not valid_candidates:
                    # Failed to construct a row. Return None or a very bad one.
                    return None
                
                if noise_level > 0 and len(valid_candidates) > 1:
                    # Pick randomly from the top few valid candidates
                    # Weight towards smaller values
                    # Simple uniform pick from top 5
                    choice = random.choice(valid_candidates[:5])
                else:
                    choice = valid_candidates[0]
                
                row.append(choice)
                # Update used_diffs
                new_diffs = {choice - a for a in row[:-1]}
                used_diffs |= new_diffs
            
            rows.append(row)
        
        # Calculate score (max entry)
        score = max(max(r) for r in rows)
        return rows, score

    def local_improve(rows, used_diffs, max_iterations=1000):
        """
        Attempts to reduce the maximum value by local moves.
        """
        rows = [r[:] for r in rows] # Deep copy
        used_diffs = set()
        for r in rows:
            used_diffs |= get_differences(r)
        
        current_score = max(max(r) for r in rows)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Find the row with the largest max value
            max_val = -1
            max_row_idx = -1
            for i, r in enumerate(rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i
            
            if max_row_idx == -1:
                break
                
            row = rows[max_row_idx]
            improved = False
            
            # Try to reduce the last element
            # Or any element that contributes to the max value?
            # The max value is always the last element since it's increasing.
            # So we try to reduce row[-1].
            
            old_val = row[-1]
            
            # Find new differences if we change row[-1] to x
            # We need to remove old diffs involving old_val and add new diffs involving x.
            
            # Iterate x from 1 to old_val - 1
            # Must be > row[-2]
            min_x = row[-2] + 1
            
            # Optimization: Only try a few candidates?
            # Let's try all valid x in (min_x, old_val)
            # But this is O(M). M is ~100. This is fine.
            
            for x in range(min_x, old_val):
                # Check if x is already in row
                if x in row:
                    continue
                
                # Calculate new diffs for the row
                # Diffs not involving last element are unchanged
                # Diffs involving last element: x - a for a in row[:-1]
                new_diffs_last = {x - a for a in row[:-1]}
                
                # Old diffs involving last element: old_val - a for a in row[:-1]
                old_diffs_last = {old_val - a for a in row[:-1]}
                
                # The set of all diffs in the system:
                # used_diffs - old_diffs_last + new_diffs_last
                # We need to ensure new_diffs_last are not in (used_diffs - old_diffs_last)
                
                # Also, new_diffs_last must not conflict with each other (they are distinct if x is distinct from a's)
                
                # Check conflict
                # The "available" diffs are those not currently used, plus the ones we are freeing up.
                # Actually, simpler:
                # Let current_used = used_diffs
                # Let freed = old_diffs_last
                # Let new = new_diffs_last
                # We need new & (current_used - freed) == empty set.
                # And new must be distinct (which is true as x is unique in row).
                
                # Note: new_diffs_last might contain values that are in freed?
                # If x - a == old_val - b, then x + b = old_val + a.
                # This is possible. If so, it's fine because we are freeing old_val - b.
                
                # Let's compute the intersection
                # Conflicting if any d in new_diffs_last is in used_diffs and d is NOT in old_diffs_last.
                
                has_conflict = False
                for d in new_diffs_last:
                    if d in used_diffs and d not in old_diffs_last:
                        has_conflict = True
                        break
                
                if not has_conflict:
                    # Valid move
                    # Update row
                    row[-1] = x
                    rows[max_row_idx] = row
                    
                    # Update used_diffs
                    used_diffs -= old_diffs_last
                    used_diffs |= new_diffs_last
                    
                    current_score = max(max(r) for r in rows)
                    improved = True
                    break # Restart loop to find new max row
            
            if not improved:
                # If we can't reduce the current max, maybe we should try a different row?
                # Or try to reduce a non-max element to allow reducing the max later?
                # For simplicity, if we can't reduce the global max, we stop this improvement phase.
                # Or we could try swapping elements between rows.
                # Let's add a simple swap heuristic:
                # Try to swap the last element of the max row with a smaller element from another row?
                # This is complex. Let's stick to single element reduction for now.
                # If stuck, break.
                break
                
        return rows, current_score

    # Main Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Increase noise as we go? Or keep it constant?
        # Start with low noise, then increase.
        noise = min(0.5, attempts * 0.01)
        
        res = greedy_construct_randomized(noise)
        if res is None:
            continue
        
        rows, score = res
        
        if score < best_score:
            best_score = score
            best_solution = rows
            # Try to improve the new best
            improved_rows, improved_score = local_improve(rows, get_differences(rows[0]), max_iterations=200)
            if improved_score < best_score:
                best_score = improved_score
                best_solution = improved_rows
        
        # If we found a very good score, we might want to do more polishing
        if best_score <= 112:
             # Do a dedicated polishing pass on the best solution found so far
             improved_rows, improved_score = local_improve(best_solution, None, max_iterations=1000)
             if improved_score < best_score:
                 best_score = improved_score
                 best_solution = improved_rows

    if best_solution is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
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

    return best_solution
