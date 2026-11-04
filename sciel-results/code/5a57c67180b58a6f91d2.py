import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = 148.0  # Safety margin before 150s kill

    num_elements = k + 1
    total_diffs_per_row = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    # Greedy construction to get a valid starting point
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Local search
    def local_search(initial_rows, max_time):
        rows = [list(r) for r in initial_rows]
        
        # Precompute initial used diffs
        used_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        current_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        # Precompute indices for diff calculation
        idx_pairs = [(i, j) for i in range(num_elements) for j in range(i+1, num_elements)]
        
        while T > min_T:
            if time.time() - start_time > max_time:
                break
                
            # Pick a random row and a random element (not the first, which is 0)
            r_idx = random.randint(0, n_rows - 1)
            e_idx = random.randint(1, num_elements - 1)
            
            old_val = rows[r_idx][e_idx]
            
            # Generate a new value
            # Bias towards small changes
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            # Enforce bounds and monotonicity
            lower = rows[r_idx][e_idx - 1] + 1
            upper = (rows[r_idx][e_idx + 1] - 1) if e_idx < num_elements - 1 else best_scope + 5 # Allow slight expansion? No, keep tight or expand if needed? 
            # Actually, if we want to minimize scope, we should try to keep it <= current_scope or best_scope.
            # Let's allow expansion up to best_scope + 10 to escape traps, but prefer smaller.
            # Wait, the goal is to minimize the MAX entry.
            # So we should try to keep all entries <= best_scope.
            # If we are stuck, maybe we need to expand temporarily? 
            # Let's stick to <= best_scope for now to enforce the constraint strictly.
            # If no move is possible, we might be stuck.
            
            # Actually, let's allow the scope to grow slightly if it helps reduce collisions, 
            # but the score is based on the final scope. 
            # The evaluator minimizes the largest entry. 
            # So we want to find a valid configuration with small max entry.
            # If we allow expansion, we might find a valid config, but with larger scope.
            # We should track the best scope found.
            
            # Let's try to keep new_val <= best_scope.
            if new_val < lower:
                new_val = lower
            if new_val > best_scope:
                new_val = best_scope
                
            if new_val == old_val:
                continue
                
            # Check collisions
            # Remove old diffs involving e_idx
            old_diffs = set()
            for i, j in idx_pairs:
                if i == e_idx or j == e_idx:
                    old_diffs.add(rows[r_idx][j] - rows[r_idx][i])
            
            # Check if new diffs collide with existing used_diffs (excluding old_diffs)
            # Or more simply:
            # 1. Compute new_diffs for the row with new_val
            # 2. Check if any new_diff is in used_diffs - old_diffs
            # 3. Check if any new_diff is duplicated within the new row (shouldn't happen if monotonic)
            
            new_row = list(rows[r_idx])
            new_row[e_idx] = new_val
            
            new_diffs = set()
            for i, j in idx_pairs:
                new_diffs.add(new_row[j] - new_row[i])
            
            # Collisions are new_diffs that are in used_diffs but NOT in old_diffs
            # Wait, used_diffs contains all diffs from all rows.
            # When we change row r_idx, the diffs from other rows remain.
            # The diffs from row r_idx change from old_diffs to new_diffs.
            # So the global set of diffs becomes:
            # (used_diffs - old_diffs) U new_diffs
            
            # Collision if:
            # 1. Any d in new_diffs is in (used_diffs - old_diffs)
            # 2. Any d in new_diffs appears more than once in new_diffs (impossible for distinct elements)
            
            collision = False
            for d in new_diffs:
                if d in used_diffs and d not in old_diffs:
                    collision = True
                    break
            
            if not collision:
                # Accept move
                used_diffs -= old_diffs
                used_diffs |= new_diffs
                rows[r_idx] = new_row
                
                new_scope = max(max(r) for r in rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                # We don't necessarily need to accept only if scope improves.
                # We accept if it's valid. The scope is just a metric we track.
                # But we want to MINIMIZE scope. 
                # So we should prefer moves that reduce scope, or at least don't increase it.
                # If a move keeps scope same or increases it, it might be a step to a better solution?
                # Usually, for this problem, smaller scope is better.
                # Let's use a cost function: cost = max_scope + (number of collisions? no, we reject collisions).
                # Since we only accept valid moves, the cost is just the scope.
                # We should accept moves that reduce scope.
                # What about moves that keep scope same? They might lead to a reduction later.
                # What about moves that increase scope? Probably bad.
                
                # Let's refine: 
                # Accept if new_scope <= best_scope.
                # If new_scope > best_scope, revert?
                # Actually, if we revert, we might get stuck.
                # But increasing scope seems counter-productive for the objective.
                # Let's enforce new_scope <= best_scope.
                
                if new_scope > best_scope:
                    # Revert
                    rows[r_idx] = list(old_val) # Wait, I lost the old row list?
                    # I need to keep old_row
                    # Let's re-implement carefully.
                    pass
            else:
                # Revert (do nothing, as we didn't modify rows yet)
                pass
                
            T *= cooling
            
        return best_rows

    # The local search above had a bug in revert logic and scope enforcement.
    # Let's write a cleaner version.
    
    def improved_local_search(initial_rows, time_limit):
        rows = [list(r) for r in initial_rows]
        used_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        current_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        idx_pairs = [(i, j) for i in range(num_elements) for j in range(i+1, num_elements)]
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
                
            r_idx = random.randint(0, n_rows - 1)
            e_idx = random.randint(1, num_elements - 1)
            
            old_val = rows[r_idx][e_idx]
            old_row = list(rows[r_idx])
            
            # Propose new value
            # Try to stay within [0, best_scope]
            # Randomly pick a new value in [lower_bound, upper_bound]
            lower = rows[r_idx][e_idx - 1] + 1
            upper = (rows[r_idx][e_idx + 1] - 1) if e_idx < num_elements - 1 else best_scope
            
            if lower > upper:
                continue
                
            # Bias towards small changes from old_val, but allow jumps
            if random.random() < 0.5:
                new_val = random.randint(lower, upper)
            else:
                delta = random.randint(-3, 3)
                new_val = old_val + delta
                if new_val < lower: new_val = lower
                if new_val > upper: new_val = upper
                
            if new_val == old_val:
                continue
                
            # Check validity
            new_row = list(rows[r_idx])
            new_row[e_idx] = new_val
            
            old_diffs = set()
            new_diffs = set()
            for i, j in idx_pairs:
                if i == e_idx or j == e_idx:
                    old_diffs.add(old_row[j] - old_row[i])
                # We need to recompute ALL diffs of the new row to check internal consistency?
                # No, internal consistency is guaranteed by strict increase.
                # We only need to check global uniqueness.
                # The global set of diffs is U = (U \ D_old) U D_new
                # Collision if D_new intersects (U \ D_old)
                
                # Wait, D_old is the set of diffs of the OLD row involving e_idx?
                # No, D_old is the set of ALL diffs of the OLD row.
                # Because changing one element changes MANY diffs.
                # Specifically, diffs involving e_idx change. Diffs NOT involving e_idx stay the same.
                # So U_new = (U \ D_old_all) U D_new_all
                # Where D_old_all is all diffs of old_row, D_new_all is all diffs of new_row.
                
                # Let's compute D_old_all and D_new_all.
                pass
            
            # Recompute logic:
            D_old = set()
            D_new = set()
            for i, j in idx_pairs:
                D_old.add(old_row[j] - old_row[i])
                D_new.add(new_row[j] - new_row[i])
                
            # Check collision:
            # Any d in D_new such that d in U and d not in D_old
            # Note: if d is in D_new and D_old, it's fine (it was already there).
            # If d is in D_new and U but NOT in D_old, it's a collision with another row.
            
            collision = False
            for d in D_new:
                if d in used_diffs and d not in D_old:
                    collision = True
                    break
            
            if not collision:
                # Valid move
                # Update used_diffs
                used_diffs -= D_old
                used_diffs |= D_new
                rows[r_idx] = new_row
                
                new_scope = max(max(r) for r in rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                # We accept the move even if scope doesn't improve, as long as it's valid.
                # This allows exploration.
                # But if scope increases, it might be bad.
                # Let's add a penalty or rejection if scope increases significantly?
                # For now, accept all valid moves.
                
            # T cooling
            T *= cooling
            
        return best_rows

    # Run greedy to get initial
    initial_rows = greedy_construct()
    initial_scope = max(max(r) for r in initial_rows)
    
    # Run local search
    # Give it most of the time
    time_limit = DEADLINE - (time.time() - start_time) - 2.0
    if time_limit < 1.0:
        return initial_rows
        
    best_rows = improved_local_search(initial_rows, time_limit)
    
    # Verify best_rows
    all_diffs = []
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.append(r[j] - r[i])
                
    if len(all_diffs) == len(set(all_diffs)):
        return best_rows
    else:
        # Fallback to initial if local search broke it (shouldn't happen)
        return initial_rows

if __name__ == "__main__":
    res = construct(7, 6)
    print(f"Scope: {max(max(r) for r in res)}")
    all_diffs = []
    for r in res:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.append(r[j] - r[i])
    print(f"Valid: {len(all_diffs) == len(set(all_diffs))}")
    print(res)
