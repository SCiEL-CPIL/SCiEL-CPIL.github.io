import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_diffs(row):
    """Compute the set of differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _get_scope(rows):
    """Compute the scope (maximum entry) of the configuration."""
    max_val = 0
    for r in rows:
        if r:
            max_val = max(max_val, r[-1])
    return max_val

def _greedy_construct(n_rows, k):
    """
    Construct an initial valid solution using a greedy approach.
    We try to pack rows one by one, choosing the smallest possible next element
    that doesn't conflict with previously used differences.
    """
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        # We need k more elements
        # We will greedily pick the smallest x > row[-1] such that
        # {x - a for a in row} is disjoint from used_diffs
        # Note: This might get stuck if we are too greedy. 
        # For initialization, it's fine.
        
        # To improve initialization, we can allow a small "jump" if stuck, 
        # but simple greedy usually works for small n_rows, k.
        # Let's use a slightly randomized greedy to avoid pathological cases.
        # Actually, let's stick to strict greedy for the initial seed, 
        # as it produces compact rows.
        
        current_max = 0
        for _ in range(k):
            # Try to find the next element
            # Start from last + 1
            candidate = row[-1] + 1
            while True:
                # Check conflicts
                new_diffs = {candidate - a for a in row}
                if not (new_diffs & used_diffs):
                    # Valid
                    row.append(candidate)
                    used_diffs |= new_diffs
                    break
                else:
                    candidate += 1
                    # Safety check: if candidate gets too large, we might have a problem
                    # But for small k, this is fast.
        
        rows.append(row)
        
    return rows

def _sa_optimize(rows, n_rows, k, time_limit=100):
    """
    Optimize the solution using Simulated Annealing.
    """
    # Deep copy
    rows = [r[:] for r in rows]
    
    # Precompute initial used_diffs
    used_diffs = set()
    for r in rows:
        used_diffs |= _compute_diffs(r)
    
    current_energy = _get_scope(rows)
    best_energy = current_energy
    best_rows = [r[:] for r in rows]
    
    # SA Parameters
    start_temp = 50.0
    end_temp = 0.1
    decay = 0.9995 # Tune this based on steps
    
    # Estimate steps
    # Each step is O(k) for validity check.
    # Let's aim for as many steps as possible within time_limit.
    
    start_time = time.time()
    deadline = start_time + time_limit
    temp = start_temp
    
    # Use a fixed seed for reproducibility? 
    # The prompt implies we should return a result. 
    # Using a fixed seed ensures the same output for the same input.
    rng = random.Random(12345)
    
    steps = 0
    while time.time() < deadline:
        # Pick a random row and a random element to move
        # We don't move index 0 (always 0)
        row_idx = rng.randint(0, n_rows - 1)
        elem_idx = rng.randint(1, k) # 1 to k inclusive
        
        row = rows[row_idx]
        old_val = row[elem_idx]
        
        # Determine valid range for new value
        low = row[elem_idx - 1] + 1
        high = row[elem_idx + 1] - 1 if elem_idx < k else current_energy + 10 # Allow expansion if needed?
        
        # Actually, to minimize scope, we usually want to move things DOWN.
        # But SA allows up.
        # Let's restrict high to current_energy to avoid infinite expansion in early steps?
        # No, if we are stuck, we might need to expand. 
        # But the scope is defined by the max. If we move a non-max element up, scope doesn't change.
        # If we move a max element up, scope increases.
        
        # Let's sample new_val from [low, high]
        # To make it efficient, let's try to move towards the center or just random.
        # A good heuristic: try to move to a random position in [low, min(high, current_energy)]
        # If we are at the boundary, maybe try to expand slightly.
        
        search_high = min(high, current_energy + 5)
        if low > search_high:
            search_high = low + 5
        
        new_val = rng.randint(low, search_high)
        
        if new_val == old_val:
            continue
            
        # Check validity of moving row[elem_idx] from old_val to new_val
        
        # 1. Remove old differences involving old_val
        # Diff with prev elements: old_val - row[i] for i in [0, elem_idx-1]
        # Diff with next elements: row[i] - old_val for i in [elem_idx+1, k]
        
        old_diffs_to_remove = set()
        for i in range(elem_idx):
            old_diffs_to_remove.add(old_val - row[i])
        for i in range(elem_idx + 1, len(row)):
            old_diffs_to_remove.add(row[i] - old_val)
        
        # 2. Calculate new differences involving new_val
        new_diffs_to_add = set()
        for i in range(elem_idx):
            new_diffs_to_add.add(new_val - row[i])
        for i in range(elem_idx + 1, len(row)):
            new_diffs_to_add.add(row[i] - new_val)
        
        # 3. Check if new_diffs_to_add conflicts with (used_diffs - old_diffs_to_remove)
        # We don't actually remove from the global set yet, we just check intersection.
        # conflict = new_diffs_to_add & (used_diffs - old_diffs_to_remove)
        # To be faster:
        # If any d in new_diffs_to_add is in used_diffs AND d is NOT in old_diffs_to_remove, it's a conflict.
        
        has_conflict = False
        for d in new_diffs_to_add:
            if d in used_diffs:
                if d not in old_diffs_to_remove:
                    has_conflict = True
                    break
        
        if has_conflict:
            # Invalid move, reject
            # Optionally, we could try to find a valid neighbor, but for now, just reject.
            # To improve acceptance rate, we could sample more carefully, but let's keep it simple.
            pass
        else:
            # Valid move. Calculate new energy.
            # Update row temporarily
            row[elem_idx] = new_val
            
            # Update used_diffs
            used_diffs -= old_diffs_to_remove
            used_diffs |= new_diffs_to_add
            
            new_energy = _get_scope(rows)
            
            # Acceptance criterion
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < pow(2.718281828, -delta / temp):
                # Accept
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_rows = [r[:] for r in rows]
            else:
                # Reject
                row[elem_idx] = old_val
                used_diffs -= new_diffs_to_add
                used_diffs |= old_diffs_to_remove
        
        # Decay temperature
        temp *= decay
        
        # Check time
        if time.time() > deadline:
            break
        
        steps += 1
        # Small sleep to avoid CPU 100%? No, we want max steps.
        
    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    # 1. Greedy Construction
    rows = _greedy_construct(n_rows, k)
    
    # 2. Simulated Annealing Optimization
    # Allocate time based on the total limit (120s).
    # Leave 5s for safety and final checks.
    time_limit = 110
    
    # If the greedy solution is already very good, we might still improve it.
    # Run SA.
    optimized_rows = _sa_optimize(rows, n_rows, k, time_limit)
    
    # Final sanity check (optional but good for debugging)
    # In production, we trust the SA.
    
    return optimized_rows

# EVOLVE-BLOCK-END
