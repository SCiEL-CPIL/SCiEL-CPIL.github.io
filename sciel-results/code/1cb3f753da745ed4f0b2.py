import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 114.0  # Safety margin from 120s limit

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1

    def row_diffs(row):
        """Calculate the set of all pairwise differences for a row."""
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs(r)
            if len(d) != needed:
                return False
            if d & used:
                return False
            used |= d
        return True

    # ---- Deterministic greedy (baseline, always valid) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # ---- Randomized greedy: diversify the myopic "smallest" choice ----
    def rand_greedy(rng, bias, window):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = []
                y = x + 1
                scanned = 0
                while len(cands) < window and scanned < 2000:
                    scanned += 1
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                while not cands:
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                if rng.random() < bias:
                    chosen = cands[0]
                else:
                    idx = int(len(cands) * (rng.random() ** 1.6))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            rows.append(row)
        return rows

    # ---- Row Reconstruction: Build a row avoiding blocked diffs ----
    def reconstruct_row(blocked, rng, max_scope):
        """
        Build a row of k+1 elements starting at 0, avoiding diffs in `blocked`.
        Returns None if not possible within max_scope.
        """
        row = [0]
        used_local = set()
        x = 0
        while len(row) < k + 1:
            found = False
            y = x + 1
            limit = max_scope
            while y <= limit:
                new = {y - a for a in row}
                if not (new & blocked) and not (new & used_local):
                    row.append(y)
                    used_local |= new
                    x = y
                    found = True
                    break
                y += 1
            if not found:
                return None
        return row

    # ---- Enhanced Simulated Annealing ----
    def local_search(start, rng, steps, T0=5.0):
        cur = [list(r) for r in start]
        rowd = [row_diffs(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d

        row_sums = [sum(r) for r in cur]
        row_maxes = [max(r) for r in cur]
        cur_sum = sum(row_sums)
        cur_scope = max(row_maxes)
        # Objective: Minimize Scope, then Sum
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0

        for it in range(steps):
            if it % 200 == 0 and remaining() < 10:
                break
            
            # Select row: bias towards rows with large max to compress them
            # 50% chance pick the row with the current global max
            if rng.random() < 0.5:
                r = row_maxes.index(cur_scope)
            else:
                r = rng.randrange(n_rows)
            
            row = cur[r]
            other_used = cur_used - rowd[r]
            
            # Move Type Selection
            move_type = rng.random()
            
            new_row = None
            nd = None
            
            if move_type < 0.4:
                # MOVE 1: Targeted Compression of the Max Element
                # Try to reduce the last element of the row
                p = len(row) - 1
                if p == 0:
                    continue
                current_max = row[p]
                prev_val = row[p - 1]
                
                # Scan downwards from current_max - 1
                # We look for a valid value that doesn't collide with other_used
                # and maintains strict increase.
                # We check a window of values to avoid O(N) scan if stuck
                found_valid = False
                # Randomize the scan start slightly to escape local traps? 
                # No, compression should be greedy downwards, but we can randomize 
                # which "candidate" we pick if multiple are valid? 
                # Actually, just finding the smallest valid one is best for compression.
                
                # Optimization: Instead of checking every single integer, 
                # we can just iterate downwards.
                # For k=7, row len is 8. Max diff is ~134. Scan is cheap.
                
                for v in range(current_max - 1, prev_val, -1):
                    # Check strict increase (guaranteed by range)
                    # Check internal diffs and external collisions
                    # We need to compute diffs for the new row
                    # new_row = row[:p] + [v]
                    
                    # Quick check: does v create a diff that is in other_used?
                    # Or does it create a duplicate diff within the row?
                    
                    # Efficient check:
                    # 1. Compute diffs of v with row[0..p-1]
                    # 2. Check if any of these are in other_used
                    # 3. Check if any of these are in rowd[r] (excluding diffs involving old row[p])
                    #    Wait, rowd[r] contains diffs involving old row[p].
                    #    We need to ensure new diffs don't collide with:
                    #    a) other_used
                    #    b) internal diffs of row[0..p-1] (which are a subset of rowd[r])
                    
                    # Let's just build the new row and compute diffs. It's O(k^2) = 120 ops.
                    # This is fast enough.
                    
                    test_row = row[:]
                    test_row[p] = v
                    
                    # Check strict increase
                    if test_row[p-1] >= v:
                        continue # Should not happen with range, but safe
                        
                    nd_test = row_diffs(test_row)
                    
                    # Check if internal diffs are unique (len == needed)
                    if len(nd_test) != needed:
                        continue
                        
                    # Check collision with other rows
                    if nd_test & other_used:
                        continue
                        
                    # Also check collision with internal diffs that *don't* involve p?
                    # No, `nd_test` is the full set of diffs for the new row.
                    # If `len(nd_test) == needed`, internal diffs are unique.
                    # The only other collision is with `other_used`.
                    # So if `nd_test & other_used` is empty, it's valid.
                    
                    new_row = test_row
                    nd = nd_test
                    found_valid = True
                    break # Greedy: take the first (smallest) valid one to maximize compression
                
                if not found_valid:
                    continue
                    
            elif move_type < 0.7:
                # MOVE 2: Random Interior Mutation
                # Pick a random index (not 0, not necessarily max) and perturb it
                p = rng.randrange(1, len(row))
                
                # Determine bounds
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 20)
                
                if lo > hi:
                    continue
                
                # Bias towards smaller values? 
                # Use a power distribution to favor smaller values in the range
                span = hi - lo + 1
                v = lo + int(span * (rng.random() ** 1.5))
                
                if v == row[p]:
                    continue
                
                test_row = row[:]
                test_row[p] = v
                
                # Check strict increase
                if (p > 0 and test_row[p-1] >= v) or (p < len(test_row)-1 and v >= test_row[p+1]):
                    continue
                    
                nd_test = row_diffs(test_row)
                if len(nd_test) != needed:
                    continue
                if nd_test & other_used:
                    continue
                
                new_row = test_row
                nd = nd_test
                
            else:
                # MOVE 3: Row Reconstruction (Structural)
                # Rebuild the row from scratch, avoiding other_used
                # Try with a tight scope first
                new_row = reconstruct_row(other_used, rng, cur_scope)
                if new_row is None:
                    # Try with a slightly looser scope to allow flexibility
                    new_row = reconstruct_row(other_used, rng, cur_scope + 10)
                if new_row is None:
                    continue
                
                nd = row_diffs(new_row)
                if len(nd) != needed or (nd & other_used):
                    continue

            # --- Acceptance Logic ---
            if new_row is None:
                continue

            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            # Update global scope
            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    # This row was the bottleneck. Check if another row is now the bottleneck
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = cur_sum - row_sums[r] + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj
            
            # Metropolis criterion
            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                # Accept
                cur[r] = new_row
                rowd[r] = nd
                cur_used = other_used | nd
                row_sums[r] = new_sum_row
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            
            # Cooling
            T = max(0.05, T * 0.9995)

        return best

    # ---- Main Search Loop ----
    # 1. Generate a good initial solution
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(20240607)

    # 2. Diversify with randomized greedy
    # We run this for a while to find a good starting point for SA
    restarts_done = 0
    while remaining() > 40 and restarts_done < 50:
        restarts_done += 1
        bias = rng.uniform(0.55, 0.95)
        window = rng.choice([4, 6, 8, 12])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        
        if s < best_scope:
            best_scope = s
            best_rows = rows
        
        # If this candidate is promising, polish it immediately
        if s < best_scope + 5 and remaining() > 25:
            pol = local_search(rows, rng, 1000, T0=3.0)
            sp = scope_of(pol)
            if sp < best_scope:
                best_scope = sp
                best_rows = pol

    # 3. Deep optimization with SA
    # We run multiple SA instances, restarting from the best or perturbed best
    sa_rounds = 0
    while remaining() > 10 and sa_rounds < 10:
        sa_rounds += 1
        
        # Occasionally perturb the best solution to escape local minima
        if sa_rounds > 1 and rng.random() < 0.3:
            # Perturb by reconstructing a random row with a looser constraint
            pert = [list(r) for r in best_rows]
            rr = rng.randrange(n_rows)
            other = set()
            for i, row in enumerate(pert):
                if i != rr:
                    other |= row_diffs(row)
            # Try to rebuild row rr with a scope slightly larger than current best
            new_row = reconstruct_row(other, rng, best_scope + 15)
            if new_row is not None:
                nd = row_diffs(new_row)
                if len(nd) == needed and not (nd & other):
                    pert[rr] = new_row
            start_sol = pert
        else:
            start_sol = best_rows
            
        # Run SA
        # Adjust steps based on time remaining
        steps = 3000
        res = local_search(start_sol, rng, steps, T0=rng.uniform(3.0, 8.0))
        
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res

    # Final safety check
    if not is_valid(best_rows):
        best_rows = base

    return best_rows
