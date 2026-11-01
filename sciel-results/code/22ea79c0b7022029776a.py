"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Greedy construction for initial solution
    def greedy_construct(n_rows, k):
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
        return rows, used
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            diffs = get_row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    # Get initial greedy solution
    rows, used = greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_score = max_entry(best_rows)
    
    if time.time() > deadline - 5:
        return best_rows
    
    # Local search: try to reduce max entry
    # Strategy: for each row, try to find a better row with smaller max
    # that doesn't conflict with other rows' differences
    
    def find_better_row(row_idx, other_rows, k, max_allowed):
        """Try to find a row with smaller max that doesn't conflict with other rows."""
        other_diffs = set()
        for i, r in enumerate(other_rows):
            if i != row_idx:
                other_diffs |= get_row_diffs(r)
        
        current_max = max(row_idx and best_rows[row_idx] or best_rows[row_idx])
        current_max = max(best_rows[row_idx])
        
        # Try to find a row with max < current_max
        for target_max in range(current_max - 1, 0, -1):
            if time.time() > deadline - 5:
                return None
            # Try random rows with this max
            for _ in range(100):
                # Generate random increasing sequence starting at 0, ending at target_max
                # Choose k-1 intermediate points
                candidates = [0]
                remaining = target_max
                for i in range(k - 1):
                    # Choose next point
                    min_next = candidates[-1] + 1
                    max_next = remaining - (k - 1 - i)  # leave room for rest
                    if min_next > max_next:
                        break
                    candidates.append(random.randint(min_next, max_next))
                    remaining = target_max - candidates[-1]
                candidates.append(target_max)
                
                if len(candidates) != k + 1:
                    continue
                if not all(candidates[i] < candidates[i+1] for i in range(k)):
                    continue
                
                diffs = get_row_diffs(candidates)
                if diffs & other_diffs:
                    continue
                # Also check within-row uniqueness is automatic for increasing seq
                return candidates
            
            # If we've tried enough random rows for this target_max, give up on it
            # (could try more systematically)
        
        return None
    
    # Iterated local search
    random.seed(42)
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        current_rows = [list(r) for r in best_rows]
        improved = False
        
        # Try to improve each row
        for row_idx in range(n_rows):
            if time.time() > deadline - 5:
                break
            
            other_rows = [current_rows[i] for i in range(n_rows) if i != row_idx]
            new_row = find_better_row(row_idx, other_rows, k, max(current_rows[row_idx]))
            
            if new_row is not None:
                # Check if this improves overall max
                new_max = max(max(r) for r in (current_rows[:row_idx] + [new_row] + current_rows[row_idx+1:]))
                if new_max < max_entry(current_rows):
                    current_rows[row_idx] = new_row
                    improved = True
                    break  # Restart from beginning
        
        if improved:
            new_max = max_entry(current_rows)
            if new_max < best_score:
                best_rows = [list(r) for r in current_rows]
                best_score = new_max
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows, _ = greedy_construct(n_rows, k)
    
    return best_rows
# EVOLVE-BLOCK-END
