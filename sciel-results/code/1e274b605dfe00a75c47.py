import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    L = k + 1  # length of each row

    def scope_of(rows):
        return max(max(r) for r in rows)

    def get_row_mask(row):
        """Compute the bitmask of differences for a single row."""
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def pure_greedy():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                # Calculate new differences if we add x
                new_diffs_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                if valid:
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
        return rows

    def randomized_greedy(window=80):
        """Greedy with shuffled candidate window using bitmask checks."""
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                for cand in cands:
                    new_diffs_mask = 0
                    valid = True
                    for a in row:
                        d = cand - a
                        bit = 1 << d
                        if used_mask & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                    if valid:
                        row.append(cand)
                        used_mask |= new_diffs_mask
                        x = cand
                        found = True
                        break
                if not found:
                    # Fallback: sequential scan
                    cand = x + 1
                    while True:
                        new_diffs_mask = 0
                        valid = True
                        for a in row:
                            d = cand - a
                            bit = 1 << d
                            if used_mask & bit:
                                valid = False
                                break
                            new_diffs_mask |= bit
                        if valid:
                            row.append(cand)
                            used_mask |= new_diffs_mask
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def local_search(rows, time_limit):
        """
        Local search using bitmask for fast validity checks.
        Moves: change one entry in one row.
        Objective: minimize scope.
        """
        rows = [r[:] for r in rows]
        n = len(rows)
        
        # Precompute masks for each row and the global used mask
        row_masks = [get_row_mask(r) for r in rows]
        global_mask = 0
        for m in row_masks:
            global_mask |= m
            
        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        current_scope = best_scope
        
        it = 0
        last_improvement = time.time()
        
        while time.time() < time_limit:
            it += 1
            if it % 1000 == 0 and time.time() - last_improvement > 5.0:
                # If stuck for 5 seconds, break out of this LS phase
                break

            ri = random.randrange(n)
            ci = random.randrange(1, L)
            
            old_val = rows[ri][ci]
            old_row = rows[ri]
            old_mask = row_masks[ri]
            
            # Generate candidates
            # Bias towards smaller values to reduce scope
            # We try values in [lower, upper]
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else current_scope
            
            if upper < lower:
                continue
                
            # Try a few random shifts and some local probes
            candidates = set()
            for _ in range(5):
                # Random within range
                candidates.add(random.randint(lower, upper))
            # Local probes
            for d in range(-5, 6):
                nv = old_val + d
                if lower <= nv <= upper:
                    candidates.add(nv)
            
            random.shuffle(list(candidates))
            
            moved = False
            for nv in candidates:
                if nv == old_val:
                    continue
                
                new_row = old_row[:]
                new_row[ci] = nv
                new_mask = get_row_mask(new_row)
                
                # Check internal validity (duplicates within row)
                if new_mask.bit_count() != diffs_per_row:
                    continue
                
                # Check collision with global used set (excluding this row's old mask)
                other_mask = global_mask & ~old_mask
                if new_mask & other_mask:
                    continue
                
                # Valid move!
                # Update state
                global_mask = (global_mask & ~old_mask) | new_mask
                row_masks[ri] = new_mask
                rows[ri] = new_row
                
                new_scope = scope_of(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    last_improvement = time.time()
                    current_scope = new_scope
                    moved = True
                    break
                elif new_scope == best_scope:
                    if random.random() < 0.5:
                        moved = True
                        break
            
            if not moved:
                pass

        return best_rows, best_scope

    def simulated_annealing(rows, time_limit):
        """
        SA with bitmask.
        """
        rows = [r[:] for r in rows]
        n = len(rows)
        row_masks = [get_row_mask(r) for r in rows]
        global_mask = 0
        for m in row_masks:
            global_mask |= m
            
        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        current_scope = best_scope
        
        T = 10.0
        T_min = 0.1
        alpha = 0.999
        it = 0
        last_improvement = time.time()
        
        while time.time() < time_limit:
            it += 1
            if it % 5000 == 0:
                T *= alpha
                if T < T_min: T = T_min
            if time.time() - last_improvement > 10.0:
                # Reheat
                T = 10.0
                last_improvement = time.time()

            ri = random.randrange(n)
            ci = random.randrange(1, L)
            
            old_val = rows[ri][ci]
            old_row = rows[ri]
            old_mask = row_masks[ri]
            
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else current_scope + 10
            
            if upper < lower:
                continue
                
            # Perturb
            shift = random.randint(-10, 5)
            nv = old_val + shift
            if nv < lower: nv = lower
            if nv > upper: nv = upper
            
            if nv == old_val:
                continue
                
            new_row = old_row[:]
            new_row[ci] = nv
            new_mask = get_row_mask(new_row)
            
            if new_mask.bit_count() != diffs_per_row:
                continue
                
            other_mask = global_mask & ~old_mask
            if new_mask & other_mask:
                continue
                
            # Valid move
            # Update state
            global_mask = (global_mask & ~old_mask) | new_mask
            row_masks[ri] = new_mask
            rows[ri] = new_row
            
            new_scope = scope_of(rows)
            
            delta = new_scope - current_scope
            if delta <= 0:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    last_improvement = time.time()
            else:
                if random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                else:
                    # Reject: revert
                    global_mask = (global_mask & ~new_mask) | old_mask
                    row_masks[ri] = old_mask
                    rows[ri] = old_row
        
        return best_rows, best_scope

    # ---------- Main Search ----------
    
    # 1. Pure Greedy
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)
    
    # 2. Randomized Greedy Restarts
    candidates = [(best_scope, best_rows)]
    
    # Time budget allocation:
    # 20% for Greedy Restarts
    # 60% for SA
    # 20% for LS Polish
    
    t_greedy_end = start_time + (deadline - start_time) * 0.2
    t_sa_end = start_time + (deadline - start_time) * 0.8
    t_ls_end = deadline
    
    attempts = 0
    while time.time() < t_greedy_end and attempts < 100:
        attempts += 1
        w = random.randint(20, 100)
        rows = randomized_greedy(window=w)
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            candidates.insert(0, (sc, rows))
            if best_scope <= 112:
                break
        elif len(candidates) < 5 and sc < best_scope + 20:
            candidates.append((sc, rows))
            
    candidates.sort(key=lambda x: x[0])
    
    # 3. Simulated Annealing on top candidates
    for i, (sc, start_rows) in enumerate(candidates[:3]):
        if time.time() >= t_sa_end - 5.0:
            break
        time_alloc = (t_sa_end - time.time()) * (0.5 if i == 0 else 0.25)
        if time_alloc < 1.0:
            continue
            
        improved, new_sc = simulated_annealing(start_rows, time_limit=time.time() + time_alloc)
        if new_sc < best_scope:
            best_rows = improved
            best_scope = new_sc
            if best_scope <= 112:
                break

    # 4. Local Search Polish
    if time.time() < t_ls_end - 2.0:
        improved, new_sc = local_search(best_rows, time_limit=t_ls_end - 2.0)
        if new_sc < best_scope:
            best_rows = improved
            best_scope = new_sc

    return best_rows
