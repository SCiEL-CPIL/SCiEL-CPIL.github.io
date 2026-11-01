import time
import random
import math
from collections import Counter

_DEADLINE = 145.0

def _row_diffs(row):
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d

def _build_counter(rows):
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return c

def _conflicts_from_counter(c):
    return sum(v - 1 for v in c.values() if v > 1)

def _scope(rows):
    return max(r[-1] for r in rows)

def _energy(rows, counter, conflicts):
    if conflicts > 0:
        return 1000 * conflicts + _scope(rows)
    else:
        return _scope(rows)

def _greedy_init(n_rows, k, rng):
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

def _sa_optimize(rows, k, deadline, rng):
    t0 = time.time()
    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    cur_conflicts = _conflicts_from_counter(cur_counter)
    cur_energy = _energy(cur_rows, cur_counter, cur_conflicts)
    
    best_rows = [list(r) for r in cur_rows]
    best_energy = cur_energy
    best_conf = cur_conflicts
    best_scope = _scope(cur_rows)
    
    # Initial temperature
    T = 50.0
    cooling = 0.99995
    min_T = 0.05
    
    n_rows = len(cur_rows)
    row_len = k + 1
    
    # Track conflicts for CDVR
    # We need to know which (row, pos) pairs are involved in conflicting differences
    # A difference d is conflicting if counter[d] > 1.
    # A position (r, p) is "bad" if it contributes to any conflicting difference.
    
    stagnation_counter = 0
    stagnation_threshold = 5000
    
    while time.time() < deadline:
        T *= cooling
        if T < min_T:
            T = min_T
            
        # CDVR: Identify rows/positions involved in conflicts
        # This is expensive if done every step, so we do it every 100 steps or when conflicts change significantly
        # For now, we use a heuristic: if conflicts > 0, pick a random conflict and a random row/pos contributing to it.
        # To speed up, we just pick a random row and pos, but bias towards "larger" values if we want to reduce scope,
        # or "conflict-heavy" if we want to fix validity.
        
        # Simple bias: 
        # If conflicts > 0, we prioritize moving elements that are part of a conflict.
        # We approximate this by just picking random, but if we fail to reduce conflicts for a while, we might restart.
        
        r_idx = rng.randrange(n_rows)
        row = cur_rows[r_idx]
        pos = 1 + rng.randrange(row_len - 1) # 1 to k
        
        old_val = row[pos]
        
        # Determine valid range for new value
        lo = row[pos - 1] + 1
        hi = (row[pos + 1] - 1) if pos + 1 < row_len else old_val + 200 # Allow growth
        
        # Propose new value
        # If we have conflicts, we might want to jump to a very different value to escape the local minimum
        # If we are valid, we want to shrink.
        
        if cur_conflicts > 0:
            # Large step to escape
            max_step = 50 + int(50 * (T / 50.0))
        else:
            # Small step to shrink
            max_step = 10 + int(10 * (T / 50.0))
            
        new_val = old_val + rng.randint(-max_step, max_step)
        if new_val < lo:
            new_val = lo
        if new_val > hi:
            new_val = hi
            
        if new_val == old_val:
            continue
            
        # Calculate delta energy
        old_d = []
        for j in range(row_len):
            if j != pos:
                if j < pos:
                    old_d.append(old_val - row[j])
                else:
                    old_d.append(row[j] - old_val)
                    
        new_d = []
        for j in range(row_len):
            if j != pos:
                if j < pos:
                    new_d.append(new_val - row[j])
                else:
                    new_d.append(row[j] - new_val)
                    
        # Update counter
        # We need to track the change in conflicts
        # delta_conflicts = change in sum(v-1 for v>1)
        
        # To do this efficiently, we check the specific diffs.
        # Let's just update the counter and re-calc conflicts for the affected diffs?
        # Actually, _conflicts_from_counter is O(N_diffs). 
        # N_diffs is at most ~200. So it's fast enough.
        
        for d in old_d:
            cur_counter[d] -= 1
            if cur_counter[d] == 0:
                del cur_counter[d]
                
        for d in new_d:
            cur_counter[d] = cur_counter.get(d, 0) + 1
            
        new_conf = _conflicts_from_counter(cur_counter)
        
        # Calculate new scope
        if pos == row_len - 1:
             other_scopes = [cur_rows[i][-1] for i in range(n_rows) if i != r_idx]
             new_scope = max([new_val] + other_scopes)
        else:
             new_scope = _scope(cur_rows)
             
        new_energy = _energy(cur_rows, cur_counter, new_conf)
        
        delta_E = new_energy - cur_energy
        
        # Metropolis
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            # Accept
            row[pos] = new_val
            cur_energy = new_energy
            cur_conflicts = new_conf
            
            # Update best
            if new_conf < best_conf or (new_conf == best_conf and new_scope < best_scope):
                best_conf = new_conf
                best_scope = new_scope
                best_rows = [list(r) for r in cur_rows]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        else:
            # Reject, revert counter
            for d in new_d:
                cur_counter[d] -= 1
                if cur_counter[d] == 0:
                    del cur_counter[d]
            for d in old_d:
                cur_counter[d] = cur_counter.get(d, 0) + 1
            # cur_rows[r_idx][pos] is unchanged
            
        # Stagnation handling: if we haven't improved best for a long time, restart from best with perturbation
        if stagnation_counter > stagnation_threshold and best_conf == 0:
             # Perturb best
             cur_rows = [list(r) for r in best_rows]
             cur_counter = _build_counter(cur_rows)
             cur_conflicts = _conflicts_from_counter(cur_counter)
             cur_energy = _energy(cur_rows, cur_counter, cur_conflicts)
             
             # Apply random perturbations
             for _ in range(3):
                 ri = rng.randrange(n_rows)
                 pos = rng.randint(1, row_len - 1)
                 lo = cur_rows[ri][pos - 1] + 1
                 hi = (cur_rows[ri][pos + 1] - 1) if pos + 1 < row_len else best_scope + 50
                 if hi > lo:
                     cur_rows[ri][pos] = rng.randint(lo, hi)
                     
             cur_counter = _build_counter(cur_rows)
             cur_conflicts = _conflicts_from_counter(cur_counter)
             cur_energy = _energy(cur_rows, cur_counter, cur_conflicts)
             
             # Reset temperature
             T = 20.0
             stagnation_counter = 0
             
             # If perturbation made it invalid, that's okay, SA will try to fix it
             
    return best_rows

def construct(n_rows, k):
    rng = random.Random(12345)
    
    # 1. Generate a valid starting point using greedy
    best_start = _greedy_init(n_rows, k, rng)
    
    # 2. Optimize with SA
    optimized_rows = _sa_optimize(best_start, k, _DEADLINE, rng)
    
    # 3. Final verification
    if not _valid(optimized_rows):
        # Fallback to greedy if SA broke it (should not happen if best tracking is correct)
        return best_start

    return optimized_rows

def _valid(rows):
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True
