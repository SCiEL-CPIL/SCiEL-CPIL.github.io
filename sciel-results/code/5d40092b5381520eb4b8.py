import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct_row(used_diffs, max_scope, seed_val):
        """
        Construct a single row of k+1 integers starting at 0, strictly increasing,
        such that all differences are in [1, max_scope] and not in used_diffs.
        Returns the row if successful, None otherwise.
        """
        random.seed(seed_val)
        row = [0]
        used_local = set()
        
        # We need to pick k elements.
        # Strategy: Greedy with randomization.
        # To ensure we can fit, we should pick elements that are "spread out" but not too much.
        
        for _ in range(k):
            last = row[-1]
            found = False
            
            # Try to find a candidate x such that x - a is not in used_diffs U used_local for all a in row.
            # And x <= max_scope.
            
            # Search window: from last+1 to max_scope.
            # To make it efficient, we can sample candidates.
            
            # Heuristic: Prefer candidates that are "close" to last to keep scope small? 
            # Or "far" to avoid small differences? 
            # Actually, we want the *row* to be compact. So we want x to be as small as possible?
            # No, if we pick x too small, we might block future picks.
            # Let's try a mix: random search in [last+1, max_scope].
            
            # Optimization: If we are close to max_scope, we must be careful.
            
            candidates = []
            # Sample some candidates
            for _ in range(50):
                x = random.randint(last + 1, max_scope)
                candidates.append(x)
            
            # Also try sequential small steps
            for x in range(last + 1, min(last + 20, max_scope + 1)):
                candidates.append(x)
                
            random.shuffle(candidates)
            
            for x in candidates:
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in used_local:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_local.update(new_diffs)
                    found = True
                    break
            
            if not found:
                # If random failed, try a more systematic search with backtracking? 
                # For now, fail this row attempt.
                return None
        
        return row

    def greedy_construct_full(seed_val):
        """Construct all rows using greedy with a compact strategy."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            # Determine a max_scope for this row.
            # We want the global scope to be small. 
            # Let's aim for a scope that is slightly larger than the lower bound.
            # The lower bound is on the *number* of differences, not the max value.
            # Max value M must satisfy M >= total_diffs.
            # So M_min = total_diffs.
            
            # Let's try to construct with M = lb + random_offset
            # But we don't know the final M. 
            # Let's just try to construct with a generous M first, then polish.
            
            # Better: Use a dynamic M. Start with lb, increase if fail.
            # But this is slow.
            
            # Alternative: Construct with M = 200 (or some heuristic upper bound).
            # The best known is 134. lb is 112.
            # Let's try M = 150.
            
            M_try = 160
            
            row = None
            attempts = 0
            while row is None and attempts < 100:
                row = greedy_construct_row(used, M_try, seed_val + attempts)
                attempts += 1
                
                if row is None and attempts % 10 == 0:
                    M_try += 5 # Expand search space if stuck
            
            if row is None:
                return None, float('inf')
            
            # Add row's diffs to used
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope

    # 1. Get initial solution
    best_rows = None
    best_scope = float('inf')
    
    for s in range(50):
        rows, scope = greedy_construct_full(s)
        if rows and scope < best_scope:
            best_scope = scope
            best_rows = rows
        if time.time() > deadline - 100:
            break
    
    if best_rows is None:
        # Fallback: Simple greedy with large search space
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)

    # 2. Row-Regeneration Simulated Annealing
    current_rows = [list(r) for r in best_rows]
    current_scope = max(max(r) for r in current_rows)
    
    # Maintain a set of all used differences for quick lookup
    # But since we regenerate whole rows, we just need to know which diffs are taken by OTHER rows.
    
    def get_used_by_others(row_idx):
        used = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            r = current_rows[i]
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    used.add(r[b] - r[a])
        return used

    best_rows_sa = [list(r) for r in current_rows]
    best_scope_sa = current_scope
    
    T = 50.0
    T_min = 1.0
    cooling = 0.999
    
    steps = 0
    last_improvement = 0
    
    while time.time() < deadline - 10:
        if T < T_min:
            T = 20.0 # Reheat
            if steps - last_improvement > 5000:
                break
        
        # Pick a row to regenerate
        r_idx = random.randint(0, n_rows - 1)
        
        # Get differences used by other rows
        used_by_others = get_used_by_others(r_idx)
        
        # Try to construct a new row for r_idx
        # We want to minimize the scope. 
        # Current scope is S. We want a row with max element <= S (or smaller).
        # Let's try to construct with max element <= current_scope.
        
        new_row = None
        # Try a few times with random seeds
        for attempt in range(10):
            # We can try to constrain the max value to be slightly smaller than current_scope
            # to encourage improvement.
            target_max = current_scope
            if random.random() < 0.5:
                target_max = max(lb, current_scope - random.randint(1, 5))
            
            new_row = greedy_construct_row(used_by_others, target_max, random.randint(0, 10000))
            if new_row:
                break
        
        if new_row:
            new_scope = max(max(r) for r in current_rows if r != current_rows[r_idx])
            new_scope = max(new_scope, max(new_row))
            
            delta = new_scope - current_scope
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_rows[r_idx] = new_row
                current_scope = new_scope
                
                if current_scope < best_scope_sa:
                    best_scope_sa = current_scope
                    best_rows_sa = [list(r) for r in current_rows]
                    last_improvement = steps
        else:
            # Failed to construct a valid row with the same/smaller scope.
            # We could try with a larger scope, but that's unlikely to improve the best.
            # So we just reject the move.
            pass
        
        T *= cooling
        steps += 1

    # 3. Final Polish
    # Try to reduce the scope by checking if we can swap out the largest element of any row
    # for a smaller one that fits in the "hole".
    
    final_rows = [list(r) for r in best_rows_sa]
    final_scope = max(max(r) for r in final_rows)
    
    # Iterate a few times
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for r_idx in range(n_rows):
            row = final_rows[r_idx]
            if len(row) < 2:
                continue
            
            last_val = row[-1]
            if last_val <= 1:
                continue
            
            # Try to replace last_val with a smaller value
            # Remove last_val from consideration
            row_without_last = row[:-1]
            
            # Get diffs used by others AND by row_without_last
            used_by_others = get_used_by_others(r_idx)
            for i in range(len(row_without_last)):
                for j in range(i + 1, len(row_without_last)):
                    used_by_others.add(row_without_last[j] - row_without_last[i])
            
            # Find smallest x < last_val such that x - a is not in used_by_others for all a in row_without_last
            # And x > row_without_last[-1]
            
            found = False
            for x in range(row_without_last[-1] + 1, last_val):
                valid = True
                for a in row_without_last:
                    d = x - a
                    if d in used_by_others:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    row.sort()
                    improved = True
                    break
            
            if found:
                # Update scope if improved
                new_scope = max(max(r) for r in final_rows)
                if new_scope < final_scope:
                    final_scope = new_scope
                    best_scope_sa = new_scope
                    best_rows_sa = [list(r) for r in final_rows]
        
        if not improved:
            break
            
    # Validate
    if is_valid(best_rows_sa):
        return best_rows_sa
    else:
        # Fallback to valid greedy
        return best_rows

