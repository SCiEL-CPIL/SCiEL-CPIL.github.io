import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs + 1
    
    # Helper: Get all diffs for a row
    def get_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]

    # Helper: Check if a row is valid given a set of used diffs
    # Returns True if valid, and updates used_diffs if valid? No, just check.
    def is_row_valid(row, used_diffs):
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        local_diffs = get_diffs(row)
        # Check for internal duplicates (should be none if strictly increasing, but safe)
        if len(local_diffs) != len(set(local_diffs)):
            return False
        # Check against global used
        for d in local_diffs:
            if d in used_diffs:
                return False
        return True

    # 1. Initial Construction: Greedy to get a valid starting point
    # We want a valid solution first. Greedy with randomization helps avoid bad local traps.
    rng = random.Random(42)
    
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                # Try random candidates in a reasonable range, then fallback to sequential
                # To keep scope somewhat small initially, bias towards smaller values
                lo = row[-1] + 1
                # Estimate hi: we don't know exact bound, but let's try up to current max + 10 or similar
                # Actually, just try sequential from lo until valid. 
                # To add randomness, shuffle the order of checking? 
                # No, sequential is deterministic. Let's add a random offset start?
                # Better: Try a few random candidates first, then sequential.
                attempts = 0
                while not found:
                    if attempts < 5:
                        # Random candidate
                        x = lo + int(rng.random() * 50)
                        if x > 200: x = lo # Cap random jump
                    else:
                        x = lo
                        lo += 1 # Sequential fallback
                        if lo > 500: return None # Safety
                        x = lo - 1 # x is the new lo? No.
                        # Let's just do sequential if random fails
                        x = row[-1]
                        while True:
                            x += 1
                            ok = True
                            for a in row:
                                if (x - a) in used:
                                    ok = False
                                    break
                            if ok:
                                break
                        # If we got here via sequential, x is valid
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
                    # Check random x
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                    else:
                        attempts += 1
            rows.append(row)
        return rows

    rows = greedy_init()
    if rows is None:
        # Fallback: Pure sequential
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)

    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Simulated Annealing to minimize scope
    # State: rows
    # Move: Pick a random row, pick a random element index > 0, try to move it to a smaller valid position.
    # If moving it makes the row invalid (order or diff conflict), reject.
    # If valid, calculate new scope.
    # Accept if new_scope < old_scope or with probability exp((old_scope - new_scope)/T)
    # Note: Scope is global max. Moving one element might not change global scope unless it was the max or unblocks others.
    # Better objective: Minimize sum of squares of elements? Or just max element?
    # Let's stick to max element (scope) as the primary metric, but use a "potential" function for SA.
    # Potential = scope + small_penalty * sum(elements) to encourage overall compactness.
    
    def get_potential(rows):
        scope = max(max(r) for r in rows)
        # Sum of all elements encourages tighter packing
        s = sum(sum(r) for r in rows)
        return scope * 1000 + s

    current_potential = get_potential(rows)
    current_scope = best_scope
    
    T = 100.0
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline - 5:
        # Pick a random row and a random position to try to reduce
        r_idx = rng.randint(0, n_rows - 1)
        # Position 0 is fixed at 0. Try positions 1..k
        pos = rng.randint(1, k)
        
        old_val = rows[r_idx][pos]
        # Try to move to a smaller value
        # New value must be > rows[r_idx][pos-1] and < rows[r_idx][pos+1] (if exists)
        lo = rows[r_idx][pos-1] + 1
        hi = (rows[r_idx][pos+1] - 1) if pos + 1 < m else old_val - 1
        
        if lo >= old_val:
            continue # Already at minimum possible for this pos given neighbors? 
                     # Actually, we can only decrease, so if lo >= old_val, we can't decrease.
                     # Wait, if lo == old_val, we are at the lower bound.
                     # If lo > old_val, it's invalid state (shouldn't happen).
        
        # Choose a new value in [lo, old_val - 1]
        if lo < old_val:
            new_val = rng.randint(lo, old_val - 1)
        else:
            continue
            
        # Check validity of new_val in this row
        # 1. Order: lo <= new_val <= hi ensures order.
        # 2. Diffs: 
        row = rows[r_idx]
        # Remove old diffs involving this pos
        old_diffs = set()
        for i in range(len(row)):
            if i != pos:
                d = abs(row[pos] - row[i])
                old_diffs.add(d)
        
        # Check if new diffs conflict with existing global diffs (excluding those from this row that involve pos)
        # We need a global used set for efficiency? 
        # Rebuilding used set every step is O(N^2). 
        # Let's maintain a global used_diffs set.
        
        # Optimization: Maintain global used_diffs
        # This requires careful update/rollback.
        
        # Let's do a full check for simplicity first, if it's too slow, we optimize.
        # For n_rows=4, k=7, N=32 elements. 
        # Checking one move: 
        # 1. Compute diffs for the modified row.
        # 2. Check if any of these diffs are in the "other rows" diffs.
        
        # Pre-compute diffs for other rows? 
        # That's static between moves on other rows.
        
        # Let's maintain a list of diff-sets for each row.
        # And a global set of all diffs.
        
        # Refactor state to support this.
        pass # This block is a placeholder for thought, actual code below implements the efficient version.

    # --- Efficient Implementation ---
    
    # Re-initialize state with efficiency in mind
    rows = [r[:] for r in best_rows]
    row_diffs = []
    global_diffs = set()
    for r in rows:
        ds = get_diffs(r)
        row_diffs.append(set(ds))
        global_diffs.update(ds)
    
    current_scope = max(max(r) for r in rows)
    current_potential = current_scope * 1000 + sum(sum(r) for r in rows)
    
    T = 50.0
    cooling_rate = 0.9995
    
    while time.time() < deadline - 5:
        r_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        
        row = rows[r_idx]
        old_val = row[pos]
        
        lo = row[pos-1] + 1
        hi = (row[pos+1] - 1) if pos + 1 < m else old_val - 1
        
        if lo >= old_val:
            continue
        
        # Try a few random new values in [lo, old_val - 1]
        # To make it faster, just pick one random value
        new_val = rng.randint(lo, old_val - 1)
        
        # Check validity
        # 1. Compute new diffs for this row involving pos
        new_local_diffs = set()
        valid = True
        for i in range(len(row)):
            if i == pos:
                continue
            d = abs(new_val - row[i])
            new_local_diffs.add(d)
            
        # 2. Check against global_diffs minus old_local_diffs of this row
        # We need to ensure new_local_diffs do not intersect with (global_diffs - row_diffs[r_idx])
        # And new_local_diffs must be unique within themselves (they are, since row is increasing and new_val is distinct)
        
        # Efficient check:
        # For each d in new_local_diffs:
        #   if d in global_diffs and d not in row_diffs[r_idx]:
        #       valid = False
        
        for d in new_local_diffs:
            if d in global_diffs:
                # Is it from this row?
                if d not in row_diffs[r_idx]:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # Valid move. Calculate new potential
        # Update scope:
        # If old_val was the unique maximum, scope might decrease.
        # If new_val is larger than current scope, scope increases (unlikely since new_val < old_val).
        
        # Calculate new sum
        delta_sum = new_val - old_val
        new_potential = current_potential + delta_sum
        
        # Scope change:
        # If old_val == current_scope:
        #   Check if any other element is current_scope.
        #   If yes, new_scope = current_scope (unless new_val > current_scope, which is false).
        #   If no, new_scope = max(second_max, new_val).
        # Else:
        #   new_scope = current_scope (since new_val < old_val <= current_scope)
        
        if old_val == current_scope:
            # Find max of all other elements
            max_other = 0
            for r_i, r in enumerate(rows):
                for p_i, v in enumerate(r):
                    if r_i == r_idx and p_i == pos:
                        continue
                    if v > max_other:
                        max_other = v
            new_scope = max(max_other, new_val)
            # Potential includes scope * 1000
            # So delta_potential = (new_scope - current_scope) * 1000 + delta_sum
            new_potential = (new_scope - current_scope) * 1000 + delta_sum + current_potential
        else:
            new_scope = current_scope
            # new_potential is already calculated as current_potential + delta_sum
            
        # Acceptance
        delta_potential = new_potential - current_potential
        if delta_potential < 0 or (T > 0 and rng.random() < math.exp(-delta_potential / T)):
            # Accept
            # Update state
            row[pos] = new_val
            
            # Update diffs
            old_row_diffs = row_diffs[r_idx]
            new_row_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    new_row_diffs.add(row[j] - row[i])
            
            # Update global_diffs
            # Remove old, add new
            # global_diffs = (global_diffs - old_row_diffs) | new_row_diffs
            
            # Efficient update:
            # For d in old_row_diffs:
            #    if d not in new_row_diffs:
            #       if no other row has d: remove from global? 
            #       This is hard without counts.
            
            # Let's just rebuild global_diffs if we accept. 
            # N=32, diffs=112. Rebuilding is O(112). Very fast.
            global_diffs = set()
            for rd in row_diffs:
                global_diffs.update(rd)
            row_diffs[r_idx] = new_row_diffs
            global_diffs.update(new_row_diffs)
            
            current_potential = new_potential
            if new_scope < current_scope:
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
