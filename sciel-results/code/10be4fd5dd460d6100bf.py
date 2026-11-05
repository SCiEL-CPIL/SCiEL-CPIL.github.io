import random
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Build a solid initial solution using a fast greedy algorithm.
    #    We use "randomized greedy" to get a diverse start.
    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We want to pick elements that keep the max entry small.
            # Standard greedy: pick smallest x > row[-1] such that no diffs collide.
            # To improve diversity and potentially escape local traps, we can randomize slightly,
            # but for the initial seed, pure greedy (smallest x) is fast and robust.
            # Actually, pure greedy tends to produce similar rows. 
            # Let's use a slight randomization in the *choice* among valid small candidates?
            # No, let's stick to deterministic greedy for the base, then improve.
            # Wait, if I run deterministic greedy for row 2, it might be identical to row 1?
            # No, because `used` contains diffs from row 1, so row 2 must avoid them.
            # So they will be different.
            
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # Helper to compute max of rows
    def get_max(rows):
        return max(max(r) for r in rows)

    # Helper to compute diffs of a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # 2. Local Search: Simulated Annealing / Hill Climbing
    #    Move: Pick a row, remove it, rebuild it with a different strategy.
    #    Strategy: 
    #       - Greedy Rebuild: Rebuild from scratch using greedy, but maybe with different "bias".
    #       - Perturbation: Modify existing row slightly? Hard to validate. 
    #       - Rebuild is safer. 
    #       - To improve on greedy, we can try "Look-ahead" or "Randomized Greedy".
    
    # We will use a "Rebuild with Constraint" approach.
    # When rebuilding row i, we have a set of forbidden diffs (used by others).
    # We want to find a row with small max entry that avoids these.
    
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit):
        """
        Tries to build a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry < max_val_limit.
        Returns the row or None.
        """
        # We use a randomized greedy approach to allow different structures.
        # We pick the next element x by looking at candidates in range [last+1, max_val_limit].
        # We prefer small x, but add noise to avoid getting stuck in the same pattern.
        
        row = [0]
        local_used = set() # Diffs within this new row
        
        for _ in range(k):
            last = row[-1]
            found = False
            # We search for a valid x.
            # To be fast, we can iterate x from last+1 to max_val_limit.
            # We collect a few valid candidates and pick one randomly (weighted towards small).
            
            valid_candidates = []
            # We don't need to go too far if we find some early.
            # But we must ensure we can finish the row.
            
            # Heuristic: if we pick x too large, we might run out of room for remaining elements?
            # No, we just need k+1 elements. As long as x <= max_val_limit, it's valid.
            
            # Let's iterate x.
            # To speed up, we can break if we have enough candidates.
            # But we want small x. So we check in increasing order of x.
            
            # Optimization: 
            # If we just do greedy (smallest x), it's deterministic given forbidden_diffs.
            # To get diversity, we can randomly skip the first few valid x's?
            # Or pick randomly from the first N valid x's.
            
            search_limit = max_val_limit
            # We'll look for the first few valid candidates.
            # Let's say we collect up to 5 valid candidates.
            
            for x in range(last + 1, search_limit + 1):
                new_diffs = {x - a for a in row}
                # Check against local_used and forbidden_diffs
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 5:
                        break
            
            if not valid_candidates:
                return None
            
            # Pick one. 
            # To encourage small max, we should bias towards smaller x.
            # Let's pick the first one (greedy) with some probability, 
            # or a random one from the top few.
            # If we always pick the smallest, we converge to the greedy solution.
            # We want to explore. 
            # Let's pick randomly from the valid_candidates found so far.
            # Since they are in increasing order of x, this biases towards small x but allows larger ones.
            
            x, new_diffs = valid_candidates[rng.randint(0, len(valid_candidates)-1)]
            
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    # Main Loop
    # We will try to reduce the current max.
    # We use a "temperature" or just a hard limit: 
    # We only accept a move if it reduces the global max.
    # If we can't reduce max, we might accept a neutral move to escape local minima?
    # Given time constraints, let's stick to "Accept if better, else reject" (Hill Climbing) 
    # but with multiple restarts if stuck? 
    # Actually, randomized rebuilds provide the diversity.
    
    rng = random.Random(12345)
    
    # To allow escaping local minima, we can allow moves that keep max same but change structure,
    # hoping to open up better moves later. 
    # However, "Reduce Max" is the primary goal.
    # Let's try: 
    # 1. Try to find a row that reduces max.
    # 2. If none found, try to find a row that is different (neutral move) to shuffle.
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        old_row = rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove from used_diffs
        used_diffs.difference_update(old_diffs)
        
        # Try to rebuild
        # Limit for new row: we hope to be <= current_max.
        # If we want to improve, we try to find a row with max < current_max.
        # If that fails, we try to find any valid row (max <= current_max) to potentially reshuffle.
        
        new_row = None
        # First, try to improve (find row with max < current_max)
        # We can binary search on max_val or just try current_max - 1.
        # If current_max is the bottleneck, reducing it is key.
        
        # Let's try to build a row with max strictly less than current_max.
        # If that fails, build with max <= current_max.
        
        # 1. Attempt to improve
        target_max = current_max - 1
        if target_max > k: # Must be at least k
            new_row = rebuild_row(k, used_diffs, rng, target_max)
            
        # 2. If failed, attempt to maintain (neutral move)
        if new_row is None:
            new_row = rebuild_row(k, used_diffs, rng, current_max)
            
        if new_row is None:
            # Rebuild failed completely. Restore old row.
            used_diffs.update(old_diffs)
            continue
        
        new_diffs = get_diffs(new_row)
        
        # Verify no conflicts (should be true by construction)
        if new_diffs & used_diffs:
            # Bug in rebuild? Restore.
            used_diffs.update(old_diffs)
            continue
            
        # Calculate new global max
        # Temporarily replace row i
        rows[i] = new_row
        new_global_max = get_max(rows)
        
        if new_global_max < current_max:
            # Improvement!
            used_diffs.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in rows]
            no_improve_count = 0
        else:
            # No improvement (or worse, though we capped at current_max so it shouldn't be worse)
            # Accept with some probability? 
            # If we always reject, we might get stuck.
            # But our rebuild is randomized.
            # Let's accept neutral moves with probability 0.1 to allow shuffling.
            if rng.random() < 0.1:
                used_diffs.update(new_diffs)
                # current_max stays same
                no_improve_count += 1
            else:
                # Reject: restore old row
                rows[i] = old_row
                used_diffs.update(old_diffs)
                no_improve_count += 1
                
        # If we are stuck for a long time, we could restart with a different seed or strategy?
        # For now, we just keep trying.
        
        # If no_improve_count is very high, we might be in a local minimum.
        # We could try a "big jump": rebuild multiple rows at once?
        # Or just accept that we've hit a local optimum for this strategy.
        # Given the 120s limit, we just keep churning.
        
        # Optional: If current_max is close to theoretical lower bound, we might be done.
        # Lower bound: n_rows * k(k+1)/2
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    return best_rows

# EVOLVE-BLOCK-END
