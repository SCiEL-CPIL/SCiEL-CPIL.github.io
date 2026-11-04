"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_rows = None
    best_max = float('inf')
    
    def try_construct(seed):
        """
        Attempt to construct a valid set using randomized backtracking.
        Returns (rows, max_entry) or (None, inf) if failed.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        # We build rows one by one.
        # For each row, we use a backtracking search to find a sequence of k+1 integers
        # starting at 0, such that all differences are distinct from 'used'.
        # To keep it fast, we limit the search depth/width or use a greedy-with-randomness approach.
        
        def build_row(target_len):
            """
            Build a single row of length target_len.
            Returns list of integers or None if backtracking fails (we don't backtrack across rows 
            in this simple version, but we randomize the choice of next number).
            """
            row = [0]
            current_max = 0
            
            # We want to find k more numbers.
            # Instead of pure greedy (smallest available), we try a few smallest available 
            # and pick one at random, or use a limited look-ahead.
            # Given time constraints, a simple randomized greedy with restarts is often effective.
            
            for _ in range(k):
                candidates = []
                # Look for the next number x > current_max
                # We scan from current_max + 1 upwards
                # To speed up, we can skip numbers that clearly conflict.
                
                x = current_max + 1
                # We will collect a few valid candidates and pick one
                found = False
                # Limit search depth for candidates to avoid long scans if stuck
                # But usually the next valid number is close.
                
                # Optimization: Pre-calculate differences to existing row elements
                # new_diffs = {x - a for a in row}
                
                # Let's just scan x incrementally
                while True:
                    if time.time() > deadline:
                        return None
                    
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        # We can stop after finding a few candidates to add randomness
                        if len(candidates) >= 5:
                            break
                    x += 1
                    # Safety break if x gets too large (shouldn't happen often if used is sparse)
                    if x > 1000:
                        break
                
                if not candidates:
                    return None
                
                # Pick a random candidate from the found ones
                # Bias towards smaller numbers to keep max small?
                # Yes, prefer smaller.
                # Sort candidates and pick from the first few
                candidates.sort()
                # Pick from the first min(3, len(candidates)) to balance optimality and randomness
                pick_from = candidates[:min(3, len(candidates))]
                chosen = rng.choice(pick_from)
                
                row.append(chosen)
                current_max = chosen
                # Update global used set
                used.update({chosen - a for a in row[:-1]})
                # Note: row[:-1] are the previous elements. The new diffs are chosen - prev.
                # Wait, the loop above calculated new_diffs as {x - a for a in row}.
                # When we append, we need to add those diffs to 'used'.
                # The code above updates used inside the while loop? No, I put it after.
                # Let's fix the update logic.
                # The 'new_diffs' calculated in the while loop for the chosen 'x' are the ones to add.
                # But I overwrote 'new_diffs' in the loop.
                # Let's re-calculate or store it.
                
            return row
        
        # The logic above had a flaw in updating 'used' correctly inside the nested function scope 
        # and the loop structure. Let's rewrite a cleaner version of the row builder.
        
        def build_row_v2(target_len):
            row = [0]
            # used is from outer scope, needs nonlocal or passed in? 
            # In Python, sets are mutable, so we can modify it directly.
            
            for _ in range(k):
                found_candidates = []
                x = row[-1] + 1
                # Scan for valid x
                # To avoid infinite loops, set a bound
                limit = row[-1] + 100 
                while x < limit:
                    if time.time() > deadline:
                        return None
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        found_candidates.append((x, new_diffs))
                        if len(found_candidates) >= 3:
                            break
                    x += 1
                
                if not found_candidates:
                    return None
                
                # Select one
                # Prefer smaller x
                found_candidates.sort(key=lambda t: t[0])
                choice = found_candidates[0]
                # Add some randomness: with prob 0.2 pick second best if exists
                if len(found_candidates) > 1 and rng.random() < 0.2:
                    choice = found_candidates[1]
                
                x, diffs = choice
                row.append(x)
                used.update(diffs)
            
            return row

        # Main loop for rows
        for _ in range(n_rows):
            if time.time() > deadline:
                return None, float('inf')
            
            row = build_row_v2(k + 1)
            if row is None:
                return None, float('inf')
            rows.append(row)
        
        max_entry = max(max(r) for r in rows)
        return rows, max_entry

    # Try the deterministic greedy first (seed 0) as a baseline
    # Actually, the seed program is pure greedy. Let's try a few seeds.
    
    # First, let's run a quick greedy to establish a baseline max
    # The original code is deterministic. We can simulate it or just use seed 0.
    # But our try_construct uses randomness even with seed 0?
    # random.Random(0) is deterministic.
    
    # Let's run a batch of seeds
    seeds = [0, 1, 2, 3, 4, 5, 10, 20, 42, 123, 999]
    # Shuffle to vary order
    rng_global = random.Random()
    rng_global.shuffle(seeds)
    
    # Add more seeds dynamically if time permits
    seed_counter = 100
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        rows, max_val = try_construct(seed)
        if rows is not None and max_val < best_max:
            best_max = max_val
            best_rows = [r[:] for r in rows] # Deep copy
            # If we found a very good one, we might want to try to improve further?
            # For now, just keep the best.
            
        # If we have time, generate more seeds
        if time.time() < deadline - 10.0:
             seeds.append(seed_counter)
             seed_counter += 1
             rng_global.shuffle(seeds)
             # Remove old seeds to keep list manageable? Or just iterate.
             # The loop iterates over 'seeds'. If we append, it continues.
             # But 'for seed in seeds' in Python 3 iterates over the list object.
             # Appending to the list during iteration will continue to the new items.
             # This is dangerous if we don't want infinite loop.
             # Let's use a while loop or a separate list for new seeds.
    
    # The above for loop with append is tricky. Let's use a while loop for better control.
    
    # Reset and use a proper search loop
    best_rows = None
    best_max = float('inf')
    
    # Try deterministic greedy (seed 0 in our randomized builder might not be pure greedy 
    # because of the "pick second best with prob 0.2" logic, but with seed 0 it's fixed.
    # To ensure we have a valid baseline, we can just run the original greedy logic separately 
    # or trust that seed 0 gives a valid result.
    
    # Let's implement a pure greedy fallback to ensure we always return something valid
    # if the randomized search fails or times out.
    
    def pure_greedy():
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

    # Run pure greedy first to have a baseline
    baseline_rows = pure_greedy()
    best_rows = baseline_rows
    best_max = max(max(r) for r in baseline_rows)
    
    # Now try randomized improvements
    current_seed = 1
    # Number of attempts based on time
    # Each attempt takes some time. Let's just loop until deadline.
    
    while time.time() < deadline:
        seed = current_seed
        current_seed += 1
        
        rows, max_val = try_construct(seed)
        if rows is not None and max_val < best_max:
            best_max = max_val
            best_rows = [r[:] for r in rows]
            
            # If we hit the theoretical lower bound or close, we could stop?
            # Lower bound is roughly (n_rows * k * (k+1) / 2)^(1/2) ? No, counting bound is 112 for (4,7).
            # Total diffs = n_rows * k*(k+1)/2.
            # Max entry M must satisfy M > sqrt(2 * total_diffs)? No.
            # The differences are distinct integers in [1, M].
            # So M >= total_diffs.
            # For (4,7), total_diffs = 4 * 28 = 112.
            # So M >= 112.
            # If best_max == 112, we can stop.
            if best_max == n_rows * k * (k + 1) // 2:
                break

    return best_rows
# EVOLVE-BLOCK-END
