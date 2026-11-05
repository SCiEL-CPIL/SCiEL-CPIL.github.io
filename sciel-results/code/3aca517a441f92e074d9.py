# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Strategy:
    1. Greedy baseline to get a valid starting point.
    2. Multi-start "Difference Pressure" Greedy Construction:
       - Build rows sequentially.
       - For each row, pick the next smallest candidate that minimizes the
         "pressure" (increase in max used difference + collision penalty).
       - This naturally packs differences into the lower range.
    3. Local Search / Polish:
       - Use a fast bitset-based energy calculation.
       - Perform random local moves to reduce the scope (max entry) while
         maintaining validity (0 collisions).
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit

    # Helper: Calculate differences for a row as a set
    def get_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: Check validity and get max entry
    def check_validity(rows):
        used = set()
        max_entry = 0
        for row in rows:
            if row[-1] > max_entry:
                max_entry = row[-1]
            for d in get_diffs_set(row):
                if d in used:
                    return False, max_entry
                used.add(d)
        return True, max_entry

    # 1. Greedy Baseline
    def greedy_construct(n_rows, k, seed=0):
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
        return rows

    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)

    # 2. Multi-start Difference Pressure Greedy
    # This method tries to build rows one by one, choosing the next element
    # that minimizes the "pressure" on the difference space.
    # Pressure = (new_max_diff - current_max_diff) + collision_penalty
    
    def pressure_greedy(n_rows, k, max_scope_hint, seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        max_used_diff = 0
        
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                best_x = None
                best_score = float('inf')
                
                # Candidate range: from last+1 to max_scope_hint
                # We can optimize by checking only up to a certain bound
                # but for small k, checking up to max_scope_hint is okay.
                # To speed up, we can stop if we find a very good candidate?
                # Let's just scan.
                
                for x in range(last + 1, max_scope_hint + 1):
                    new_diffs = {x - a for a in row}
                    collisions = len(new_diffs & used)
                    
                    if collisions > 0:
                        # Heavy penalty for collisions
                        score = 1000 + collisions
                    else:
                        new_max_diff = max(new_diffs) if new_diffs else 0
                        # We want to minimize the spread of differences
                        # so that we have room for future rows.
                        # Score based on how much we extend the max diff
                        score = new_max_diff - max_used_diff
                        if score < 0:
                            score = 0 # Should not happen if x > last and diffs are positive
                            
                    if score < best_score:
                        best_score = score
                        best_x = x
                        # Early exit if we found a perfect candidate (no collision, minimal increase)
                        # If score is 0 or 1, it's very good.
                        if score <= 1:
                            break
                            
                if best_x is None:
                    return None # Failed to build within scope
                    
                row.append(best_x)
                new_diffs = {best_x - a for a in row[:-1]}
                used |= new_diffs
                if new_diffs:
                    if max(new_diffs) > max_used_diff:
                        max_used_diff = max(new_diffs)
                last = best_x
            
            rows.append(row)
            
        return rows

    # Try to improve using Pressure Greedy with various scope hints
    # Start with a scope slightly smaller than the greedy baseline
    current_best_max = best_max
    
    # Allocate time for Pressure Greedy attempts
    pg_start = time.time()
    pg_time_budget = 30.0
    
    # Try decreasing scope
    # We know the counting bound is roughly n_rows * k*(k+1)/2
    # Let's try scopes from current_best_max - 1 down to a lower bound
    lower_bound = n_rows * k * (k + 1) // 2 + 5 # Rough lower bound
    
    # We will try a few random seeds for each scope to add diversity
    scopes_to_try = range(current_best_max - 1, max(lower_bound, 0), -1)
    
    # We can't try all scopes if there are too many. 
    # Let's sample them or just go down until time runs out.
    # Given k=8, n_rows=4, current_best_max is likely ~150-200.
    # So ~50-100 scopes. Each scope takes a few ms.
    
    for scope in scopes_to_try:
        if time.time() - pg_start > pg_time_budget:
            break
            
        # Try a few seeds for this scope
        for seed in range(5):
            if time.time() - pg_start > pg_time_budget:
                break
                
            res = pressure_greedy(n_rows, k, scope, seed)
            if res is not None:
                # Verify validity just in case
                valid, m = check_validity(res)
                if valid and m < best_max:
                    best_max = m
                    best_rows = res
                    # Update current_best_max to allow tighter search later?
                    # No, we are iterating down, so we just continue.
                    # But we can break out of seed loop if we found a solution for this scope?
                    # Yes, since we are scanning down, finding one for this scope is good.
                    # But maybe a different seed finds a smaller max within the same scope?
                    # The scope is the cap. The actual max might be smaller.
                    # So we should probably keep trying seeds for this scope to see if we get a smaller max?
                    # Actually, if we find a valid solution with cap `scope`, its max is <= scope.
                    # If we find another with cap `scope`, it might have max < previous best.
                    # So we continue seeds.
                    pass
        # If we found a solution with this scope, we can break?
        # No, we want to go lower. The loop continues to scope-1.
        
    # 3. Local Search / Polish on the best found so far
    # We use a fast bitset-based energy function.
    
    # Bitset size: max difference could be up to ~200. 256 bits is enough.
    # We use Python integers as bitsets.
    
    def compute_energy_bitset(rows):
        used_bits = 0
        max_entry = 0
        collisions = 0
        
        for row in rows:
            if row[-1] > max_entry:
                max_entry = row[-1]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    bit = 1 << d
                    if used_bits & bit:
                        collisions += 1
                    else:
                        used_bits |= bit
                        
        # Energy: Collisions * 100 + max_entry
        # We prioritize validity (collisions=0) then minimize max_entry.
        return collisions * 100 + max_entry

    current_rows = [row[:] for row in best_rows]
    current_energy = compute_energy_bitset(current_rows)
    
    # SA Parameters
    T = 10.0
    alpha = 0.995
    T_end = 0.5
    rng = random.Random(123)
    
    steps = 0
    max_steps = 200000
    
    while T > T_end and steps < max_steps:
        if time.time() - start_time > time_limit - 5:
            break
            
        r = rng.randint(0, n_rows - 1)
        i = rng.randint(1, k)
        
        min_val = current_rows[r][i-1] + 1
        max_val = current_rows[r][i+1] - 1 if i < k else 10000
        
        delta = rng.randint(-3, 3)
        if delta == 0:
            delta = 1 if rng.random() < 0.5 else -1
            
        new_val = current_rows[r][i] + delta
        
        if new_val < min_val:
            new_val = min_val
        elif new_val > max_val:
            new_val = max_val
            
        if new_val == current_rows[r][i]:
            T *= alpha
            steps += 1
            continue

        old_val = current_rows[r][i]
        current_rows[r][i] = new_val
        
        new_energy = compute_energy_bitset(current_rows)
        
        dE = new_energy - current_energy
        if dE < 0 or rng.random() < math.exp(-dE / T):
            current_energy = new_energy
            steps += 1
            if current_energy < best_max: # Valid and better max (since collision*100 dominates)
                # Actually, if current_energy < best_max, it implies collisions=0 and max_entry < best_max
                # Wait, best_max is just the max entry.
                # If current_energy is e.g. 100 + 130 = 230, and best_max is 134.
                # We need to check validity explicitly or parse energy.
                # Let's just check if collisions are 0.
                # compute_energy_bitset returns collisions*100 + max_entry.
                # If collisions == 0, energy == max_entry.
                # So if current_energy < best_max, it is valid and better.
                best_max = current_energy
                best_rows = [row[:] for row in current_rows]
        else:
            current_rows[r][i] = old_val
        
        T *= alpha

    # Final check
    valid, m = check_validity(best_rows)
    if not valid:
        # Fallback to greedy
        return greedy_construct(n_rows, k)
        
    return best_rows
# EVOLVE-BLOCK-END
