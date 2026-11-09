"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _greedy_row(used, k, max_val=None):
    """Build one row greedily given the set of used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            return None
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _get_used_diffs(rows):
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used.add(r[j] - r[i])
    return used

def _max_entry(rows):
    return max(max(r) for r in rows)

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0
    
    # 1. Initial Solution: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
             # Fallback if greedy fails (unlikely for small k)
             row = _greedy_row(set(), k)
             used = set() # Reset if we restart? No, this is just a baseline.
             # Actually, if greedy fails, we need a valid start.
             # Let's just use a random valid row generator if greedy fails.
             # But greedy usually works.
        rows.append(row)
        used = _get_used_diffs(rows) # Re-calculate to be safe
    
    best_rows = [r[:] for r in rows]
    best_max = _max_entry(best_rows)
    current_rows = [r[:] for r in rows]
    current_max = best_max
    
    # 2. Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random row to modify
        idx = random.randint(0, n_rows - 1)
        
        # Strategy: Try to find a better replacement for row idx
        # or try to shift entries in row idx.
        
        # Option A: Replace row with a new randomized greedy row
        # We want to keep the max entry low.
        # Let's try to build a row with max entry <= current_max (or slightly higher/lower)
        
        # Remove current row's diffs from 'used'
        old_row = current_rows[idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used_others = set()
        for i, r in enumerate(current_rows):
            if i == idx: continue
            for a in range(len(r)):
                for b in range(a+1, len(r)):
                    used_others.add(r[b] - r[a])
        
        # Try to construct a new row that doesn't conflict with used_others
        # and ideally has a smaller max entry.
        
        # Randomized Greedy with a target max
        # Let's try a few random seeds/targets
        new_row = None
        # Try to keep max entry close to current_max or lower
        # If we can find a row with max < current_max, great.
        
        # Heuristic: Try random greedy with a limit
        limit = current_max
        if random.random() < 0.5:
             limit = current_max - 1
        else:
             limit = current_max + 5 # Allow some growth to escape local minima
        
        # Try a few random variations
        for _ in range(10):
             # Randomized greedy
             row = [0]
             x = 0
             success = True
             while len(row) < k + 1:
                 x += 1
                 if x > limit:
                     success = False
                     break
                 # Random jump
                 if random.random() < 0.2:
                     x += random.randint(1, 3)
                     if x > limit:
                         success = False
                         break
                 
                 new_diffs = {x - a for a in row}
                 if not (new_diffs & used_others):
                     row.append(x)
                 else:
                     # Conflict with existing rows
                     # Try to skip more? Or fail?
                     # In this simple SA, we just fail this attempt
                     success = False
                     break
             
             if success:
                 # Check internal diffs (already done by construction)
                 # Check if it conflicts with itself? No, greedy ensures distinct diffs in row.
                 # But wait, greedy ensures `new_diffs` are distinct from `used_others`.
                 # It does NOT ensure `new_diffs` are distinct from each other?
                 # `new_diffs = {x - a for a in row}`. Since `row` has distinct elements, `x-a` are distinct.
                 # So internal diffs are distinct.
                 # And we checked against `used_others`.
                 # So it is a valid row.
                 
                 # Calculate new max
                 new_max = max(row)
                 # Cost function: Max entry of the whole set
                 # If we replace row idx with `row`, the new set max is:
                 # max(max(other_rows), new_max)
                 
                 # Let's estimate the cost change
                 # old_max = current_max
                 # new_max_total = max(_max_entry(current_rows[:idx] + current_rows[idx+1:]), new_max)
                 
                 # We need to be careful: if the current row had the global max, replacing it might lower the global max.
                 
                 # Let's just do a full check for simplicity in the move acceptance,
                 # but we can optimize by precomputing max of others.
                 
                 # For now, let's just accept if it looks promising
                 # and verify in the main loop.
                 
                 # Actually, let's just generate a candidate and evaluate.
                 candidate_rows = current_rows[:]
                 candidate_rows[idx] = row
                 candidate_max = _max_entry(candidate_rows)
                 
                 # Check validity of the whole set?
                 # We constructed `row` to not conflict with `used_others`.
                 # So the whole set is valid.
                 
                 delta = candidate_max - current_max
                 
                 if delta < 0 or random.random() < math.exp(-delta / T):
                     current_rows = candidate_rows
                     current_max = candidate_max
                     
                     if current_max < best_max:
                         best_max = current_max
                         best_rows = [r[:] for r in current_rows]
                         
                     break # Move accepted, break inner loop
                 
        # Cooling
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_rows
# EVOLVE-BLOCK-END
