"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry: must be at least total_diffs
    lower_bound = total_diffs
    
    # Bitset size: we estimate the max possible value.
    # A safe upper bound is around 2 * lower_bound + 100
    max_est = 2 * lower_bound + 100
    if max_est < 1000:
        max_est = 1000
        
    # For bitset operations, we need to be careful with Python's arbitrary precision ints.
    # We will use integers as bitsets.
    
    def bitset_add(bs, d):
        return bs | (1 << d)
        
    def bitset_remove(bs, d):
        return bs & ~(1 << d)
        
    def bitset_has(bs, d):
        return (bs >> d) & 1

    def pure_greedy():
        """
        Generate a valid initial solution using a greedy approach.
        This ensures we start with a valid state for annealing.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    baseline_rows = pure_greedy()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in baseline_rows)
    
    # If we hit the lower bound, we can return immediately
    if best_max == lower_bound:
        return best_rows

    def rows_to_diffs(rows):
        """
        Convert rows (list of lists of ints) to a list of sorted lists of differences.
        Also returns the bitsets.
        """
        diffs_list = []
        bitsets = []
        for r in rows:
            d_list = []
            bs = 0
            for i in range(len(r)):
                for j in range(i):
                    d = r[i] - r[j]
                    d_list.append(d)
                    bs |= (1 << d)
            d_list.sort()
            diffs_list.append(d_list)
            bitsets.append(bs)
        return diffs_list, bitsets

    def diffs_to_rows(diffs_list):
        """
        Reconstruct rows from differences.
        This is the hard part. Given a set of differences for a row,
        can we find the sequence?
        
        Actually, for the annealing state, we can just store the rows directly
        and update them locally. But checking validity from rows is O(k^2).
        
        Alternative: Store the rows. 
        Moves:
        1. Shrink: Try to reduce the last element of a row.
           This changes the last element x to x'. 
           The differences changed are (x - a_i) for all i < last.
           We need to ensure new differences are not used elsewhere.
           
        2. Swap: Swap a difference in row A with a difference in row B.
           This is complex because it changes the structure of the row.
           
        Let's stick to a simpler move set that operates on the rows directly but uses bitsets for fast conflict checking.
        
        Move 1: "Shrink"
        Pick a row r. Pick the last element x.
        Try to replace x with x' < x.
        The differences involving x are D_old = {x - a_i for i in 0..k-1}.
        The new differences D_new = {x' - a_i for i in 0..k-1}.
        The differences NOT involving x remain unchanged.
        
        Conflict check:
        1. D_new must not intersect with (GlobalUsed - D_old).
           GlobalUsed is the union of all differences in all rows.
           We can maintain a global bitset `global_used`.
           Conflict if (bitset(D_new) & (global_used ^ bitset(D_old))) != 0.
           Note: global_used includes D_old. So we remove D_old from the mask.
           
        2. Within the row, the new element x' must be > a_{k-1}.
           And the differences x' - a_i must be distinct? 
           Since a_i are distinct and x' is fixed, x' - a_i are distinct.
           
        So the only check is against other rows and the rest of the same row?
        Wait, D_old are differences with previous elements.
        The differences between previous elements (a_j - a_i) are unchanged.
        So we only need to check D_new against (GlobalUsed - D_old).
        
        Move 2: "Swap" (between two rows)
        This is harder to define simply.
        
        Let's focus on "Shrink" and "Random Restart" or "Local Search" on the last element.
        Actually, a common technique for DST is to optimize the last element of each row.
        
        Let's implement a local search that tries to minimize the sum of the last elements (proxy for max).
        
        State: rows
        Cost: max(max(row) for row in rows)
        
        We will use Simulated Annealing.
        
        Moves:
        1. Pick a random row r.
        2. Pick a random position p in that row (1..k).
        3. Try to change rows[r][p] to a new value.
           If p < k, changing it affects differences with all subsequent elements in the same row?
           No, the row must be increasing. If we change rows[r][p], we must ensure:
           rows[r][p-1] < new_val < rows[r][p+1] (if p < k).
           And differences with all other elements in the row must be distinct and not used elsewhere.
           
           This is expensive to check if we don't use bitsets carefully.
           
           Let's stick to modifying only the LAST element of a row.
           This is the "Shrink" move. It is cheap to check.
           
           Move: Shrink row r.
           Current last element x.
           Try new value x' in range (rows[r][k-1] + 1, x).
           D_old = {x - a_i for i in 0..k-1}
           D_new = {x' - a_i for i in 0..k-1}
           
           Check:
           mask = global_used
           for d in D_old: mask = bitset_remove(mask, d)
           if bitset(D_new) & mask: invalid.
           
           If valid, accept with probability based on cost change.
           
        To escape local minima, we also need moves that change the structure of the row, not just the last element.
        But "Shrink" is the primary way to reduce the scope.
        
        Let's add a "Perturbation" move:
        Randomly change a middle element?
        If we change rows[r][p] to x', we must recompute all differences involving x'.
        D_old = {x - a_i for i != p}
        D_new = {x' - a_i for i != p}
        Check against global_used - D_old.
        Also check internal consistency:
        The set of differences within the row must still be valid (distinct).
        Since a_i are distinct, x' - a_i are distinct.
        But we must ensure x' - a_i are not in D_internal (differences between other a's).
        Wait, D_internal is part of global_used.
        So the check `bitset(D_new) & (global_used - D_old)` covers everything.
        
        So the move is:
        1. Pick row r, pos p.
        2. Pick new value x'.
        3. Check bounds: rows[r][p-1] < x' < rows[r][p+1] (with edges).
        4. Compute D_old, D_new.
        5. Check conflicts.
        
        This allows changing any element.
        
        Cost function:
        We want to minimize the largest entry.
        Let M = max(max(row) for row in rows).
        Cost = M.
        
        However, minimizing M directly is step-wise.
        We can use a smoother cost: sum of last elements, or sum of all elements.
        Let's use Cost = sum(rows[r][k] for r in range(n_rows)).
        This encourages shrinking all rows.
        
        """
        pass

    # Implementation of Simulated Annealing
    
    def get_cost(rows):
        # Sum of last elements is a good proxy
        return sum(rows[r][k] for r in range(n_rows))
        
    def get_max_scope(rows):
        return max(max(row) for row in rows)

    # Initial state
    current_rows = baseline_rows
    current_diffs, current_bitsets = rows_to_diffs(current_rows)
    
    # Global used bitset
    global_used = 0
    for bs in current_bitsets:
        global_used |= bs
        
    current_cost = get_cost(current_rows)
    best_rows_sa = [r[:] for r in current_rows]
    best_cost_sa = current_cost
    best_max_sa = get_max_scope(current_rows)
    
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    rng = random.Random(42)
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        # Pick a random row and position
        r = rng.randint(0, n_rows - 1)
        p = rng.randint(1, k) # 1 to k
        
        old_val = current_rows[r][p]
        
        # Determine range for new value
        # Must be > previous, < next
        min_val = current_rows[r][p-1] + 1
        max_val = current_rows[r][p+1] - 1 if p < k else best_max_sa + 10 # Allow growth if needed? 
        # Actually, we want to shrink. Let's bias towards smaller values.
        # But to explore, we should allow some range.
        
        # If we are trying to shrink, we look at [min_val, old_val]
        # If we are trying to explore, we look at [min_val, max_val]
        
        # Let's pick a random value in a window around old_val
        # Window size depends on T
        window = int(T * 0.1) + 1
        low = max(min_val, old_val - window)
        high = min(max_val, old_val + window)
        
        if low > high:
            continue
            
        new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
            
        # Compute D_old and D_new
        # D_old: differences between old_val and all other elements in row r
        # D_new: differences between new_val and all other elements in row r
        
        D_old = set()
        D_new = set()
        
        for i in range(k + 1):
            if i == p:
                continue
            D_old.add(abs(old_val - current_rows[r][i]))
            D_new.add(abs(new_val - current_rows[r][i]))
            
        # Check conflicts
        # Remove D_old from global_used
        mask = global_used
        for d in D_old:
            mask = bitset_remove(mask, d)
            
        # Check if D_new intersects with mask
        conflict = False
        for d in D_new:
            if bitset_has(mask, d):
                conflict = True
                break
                
        if conflict:
            continue
            
        # Valid move
        # Calculate cost change
        # Cost is sum of last elements.
        # Only row r's last element might change if p == k?
        # No, cost is sum of rows[r][k].
        # If p < k, rows[r][k] doesn't change directly, but the row is different.
        # Wait, if p < k, the last element is unchanged. So cost doesn't change?
        # That's a problem. If we only measure sum of last elements, moving internal elements has 0 cost change.
        # We need a cost that reflects the "tightness" or potential for shrinking.
        
        # Let's use a different cost: Sum of all elements?
        # Or just accept all valid moves if T is high, and bias towards shrinking later?
        
        # Better cost: Max scope.
        # If we change an internal element, max scope might not change immediately,
        # but it might enable shrinking the last element later.
        
        # Let's use Cost = get_max_scope(rows).
        # If new_val < old_val and p == k, cost might decrease.
        # If p < k, cost is likely unchanged.
        
        # To make SA work, we need a cost that changes with internal moves.
        # How about: Cost = sum(1 / (val - prev_val))? No.
        
        # Let's stick to Max Scope, but if the move doesn't change Max Scope, 
        # we accept it with probability 1.0 (neutral move) to allow exploration.
        # This is a common trick.
        
        old_max = get_max_scope(current_rows)
        
        # Apply move
        current_rows[r][p] = new_val
        
        # Update bitsets
        # Remove old diffs from global_used
        for d in D_old:
            global_used = bitset_remove(global_used, d)
            current_bitsets[r] = bitset_remove(current_bitsets[r], d)
            
        # Add new diffs
        for d in D_new:
            global_used = bitset_add(global_used, d)
            current_bitsets[r] = bitset_add(current_bitsets[r], d)
            
        new_max = get_max_scope(current_rows)
        
        delta_cost = new_max - old_max
        
        if delta_cost < 0:
            # Improvement
            current_cost = new_max
            best_rows_sa = [row[:] for row in current_rows]
            best_cost_sa = new_max
            if new_max < best_max_sa:
                best_max_sa = new_max
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # No improvement or worse
            if delta_cost == 0:
                # Accept neutral moves to explore
                current_cost = new_max
                no_improve_steps += 1
            else:
                # Worse
                prob = math.exp(-delta_cost / T)
                if rng.random() < prob:
                    current_cost = new_max
                    no_improve_steps += 1
                else:
                    # Revert
                    current_rows[r][p] = old_val
                    for d in D_new:
                        global_used = bitset_remove(global_used, d)
                        current_bitsets[r] = bitset_remove(current_bitsets[r], d)
                    for d in D_old:
                        global_used = bitset_add(global_used, d)
                        current_bitsets[r] = bitset_add(current_bitsets[r], d)
                    continue
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        steps += 1
        
        # Check for restart
        if no_improve_steps > max_no_improve:
            # Restart from best found so far with higher T
            current_rows = [row[:] for row in best_rows_sa]
            current_diffs, current_bitsets = rows_to_diffs(current_rows)
            global_used = 0
            for bs in current_bitsets:
                global_used |= bs
            current_cost = get_cost(current_rows)
            T = 50.0
            no_improve_steps = 0
            
    # Update global best
    if best_max_sa < best_max:
        best_rows = best_rows_sa
        best_max = best_max_sa
        
    return best_rows
# EVOLVE-BLOCK-END
