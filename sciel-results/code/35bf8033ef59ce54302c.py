import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    
    # --- Initialization: Greedy with randomization ---
    def greedy_init(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                # Start from smallest possible, but sometimes jump ahead
                # to allow diversity in the initial population
                start_val = row[-1] + 1
                if len(row) > 1 and rng.random() < 0.3:
                    start_val = row[-1] + rng.randint(2, 5)
                
                found = False
                for v in range(start_val, 200):
                    valid = True
                    new_diffs = []
                    for u in row:
                        d = v - u
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(v)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    # Fallback: standard greedy from last+1
                    v = row[-1] + 1
                    while True:
                        valid = True
                        new_diffs = []
                        for u in row:
                            d = v - u
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        if valid:
                            row.append(v)
                            used.update(new_diffs)
                            break
                        v += 1
            rows.append(row)
        return rows

    # --- Helper Functions ---
    def get_scope(rows):
        return max(r[-1] for r in rows)

    def count_collisions(rows):
        counts = Counter()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    counts[d] += 1
        coll = 0
        for c in counts.values():
            if c > 1:
                coll += c - 1
        return coll

    def calculate_cost(rows, scope, collisions):
        # Priority 1: Validity (collisions == 0)
        # Priority 2: Minimize scope
        # Weight collisions heavily to steer towards valid solutions first
        return scope + 10000 * collisions

    # --- Local Search: Row Rebuild ---
    def rebuild_row(row_idx, current_rows, diff_counts, max_scope_allowed):
        """
        Attempts to rebuild a specific row to minimize conflicts with other rows.
        Returns a new row or None if failed.
        """
        other_rows = [r for i, r in enumerate(current_rows) if i != row_idx]
        
        # Calculate differences used by other rows
        used_by_others = Counter()
        for r in other_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_by_others[r[j] - r[i]] += 1
        
        # Greedy construction of the new row
        # We want to minimize the maximum value (scope) of this row
        # while avoiding differences already used by others.
        
        best_row = None
        best_row_max = float('inf')
        
        # Try a few random starts or just the standard greedy
        # Since we are optimizing, standard greedy is a good local optimum
        row = [0]
        x = 0
        for _ in range(1, row_len):
            found = False
            # Look for the smallest x that doesn't conflict
            # Limit search space to current best scope + some slack
            search_limit = max(max_scope_allowed, best_row_max) + 10
            for v in range(x + 1, search_limit + 1):
                valid = True
                for u in row:
                    d = v - u
                    if used_by_others[d] > 0:
                        valid = False
                        break
                if valid:
                    row.append(v)
                    # Temporarily mark these diffs as used for subsequent steps in this row?
                    # No, within a row, diffs must be distinct. 
                    # But the `used_by_others` is fixed.
                    # We also need to ensure diffs within `row` are distinct.
                    # The greedy check `v - u` for all `u in row` ensures this 
                    # IF we also check against previously generated diffs in `row`.
                    # Wait, `used_by_others` doesn't include current row's internal diffs.
                    # We need a local set for the row.
                    pass
            
            # The simple greedy above is flawed because it doesn't track internal row diffs.
            # Let's implement a proper internal greedy.
            
        # Proper Rebuild
        row = [0]
        local_diffs = set()
        x = 0
        for _ in range(1, row_len):
            found = False
            search_limit = max(max_scope_allowed, best_row_max) + 10
            for v in range(x + 1, search_limit + 1):
                valid = True
                new_diffs = []
                for u in row:
                    d = v - u
                    if d in local_diffs or used_by_others[d] > 0:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(v)
                    local_diffs.update(new_diffs)
                    x = v
                    found = True
                    break
            if not found:
                return None
        
        return row

    # --- Main Search Loop ---
    # Start with a few diverse greedy inits
    best_rows = None
    best_cost = float('inf')
    
    # Try a few seeds
    for s in range(10):
        if time.time() > deadline - 5:
            break
        rows = greedy_init(seed=s)
        scope = get_scope(rows)
        coll = count_collisions(rows)
        cost = calculate_cost(rows, scope, coll)
        
        if cost < best_cost:
            best_cost = cost
            best_rows = [r[:] for r in rows]
            
    if best_rows is None:
        best_rows = greedy_init(seed=0)
        best_cost = calculate_cost(best_rows, get_scope(best_rows), count_collisions(best_rows))

    current_rows = [r[:] for r in best_rows]
    current_scope = get_scope(current_rows)
    current_coll = count_collisions(current_rows)
    current_cost = best_cost
    
    # Precompute diff counts for current state to allow fast delta updates
    # But row rebuilds change the whole row, so we might just recalculate.
    # Given small N, recalculation is fine.
    
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    rng = random.Random(42)
    
    steps = 0
    last_improve = time.time()
    
    while time.time() < deadline - 2:
        # Move 1: Rebuild a random row
        r_idx = rng.randint(0, n_rows - 1)
        
        # Determine max scope allowed for the new row to keep global scope low
        # Ideally, the new row shouldn't exceed current global scope
        # But allowing slight increases helps escape local minima
        max_allowed = current_scope + rng.randint(0, 5)
        
        new_row = rebuild_row(r_idx, current_rows, None, max_allowed)
        
        if new_row is None:
            T *= cooling
            continue
            
        # Calculate new state cost
        new_rows = [r[:] for r in current_rows]
        new_rows[r_idx] = new_row
        
        new_scope = get_scope(new_rows)
        new_coll = count_collisions(new_rows)
        new_cost = calculate_cost(new_rows, new_scope, new_coll)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / max(T, min_T)):
            current_rows = new_rows
            current_scope = new_scope
            current_coll = new_coll
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [r[:] for r in current_rows]
                last_improve = time.time()
                
        T *= cooling
        if T < min_T:
            T = min_T
        
        steps += 1
        
        # Stagnation handling: if no improvement for a long time, restart with new seed
        if time.time() - last_improve > 20 and steps > 1000:
             # Perturb current best
             r_idx = rng.randint(0, n_rows - 1)
             p_idx = rng.randint(1, row_len - 1)
             old_val = best_rows[r_idx][p_idx]
             low = best_rows[r_idx][p_idx - 1] + 1
             high = best_rows[r_idx][p_idx + 1] - 1 if p_idx < row_len - 1 else old_val + 10
             if low <= high:
                 new_val = rng.randint(low, high)
                 best_rows[r_idx][p_idx] = new_val
                 # Recalc best cost
                 best_scope = get_scope(best_rows)
                 best_coll = count_collisions(best_rows)
                 best_cost = calculate_cost(best_rows, best_scope, best_coll)
                 current_rows = [r[:] for r in best_rows]
                 current_scope = best_scope
                 current_coll = best_coll
                 current_cost = best_cost
                 last_improve = time.time()
                 steps = 0 # Reset step counter for stagnation check

    # Final verification and return
    # Ensure we return a valid solution if possible.
    # If best_rows has collisions, we should try to find a valid one or return the best invalid one?
    # The task says "Valid when all positive differences... are distinct".
    # "Score: ... minimized; it is an upper bound on m...".
    # Usually, these tasks require a VALID solution to score > 0.
    # If best_rows is invalid, we might score 0.
    # Let's check validity of best_rows.
    
    final_scope = get_scope(best_rows)
    final_coll = count_collisions(best_rows)
    
    # If we have collisions, try a quick local fix on the best_rows
    if final_coll > 0:
         # Try to fix by rebuilding rows with high contribution to collisions
         # This is a fallback
         for _ in range(5000):
             if time.time() > deadline - 1:
                 break
             r_idx = rng.randint(0, n_rows - 1)
             new_row = rebuild_row(r_idx, best_rows, None, final_scope + 10)
             if new_row is not None:
                 trial_rows = [r[:] for r in best_rows]
                 trial_rows[r_idx] = new_row
                 trial_coll = count_collisions(trial_rows)
                 trial_scope = get_scope(trial_rows)
                 if trial_coll < final_coll or (trial_coll == final_coll and trial_scope < final_scope):
                     best_rows = trial_rows
                     final_coll = trial_coll
                     final_scope = trial_scope
                     if final_coll == 0:
                         break

    return best_rows
