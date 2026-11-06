"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    best_result = []
    best_len = 0
    
    def is_valid_addition(S, x, sums_set):
        """Check if adding x to S keeps all triple sums unique."""
        n = len(S)
        
        # (x, x, x)
        s = 3 * x
        if s in sums_set:
            return False
        
        # (x, x, y) for y in S
        for y in S:
            s = 2 * x + y
            if s in sums_set:
                return False
        
        # (x, y, z) for y <= z in S
        for i in range(n):
            y = S[i]
            for j in range(i, n):
                z = S[j]
                s = x + y + z
                if s in sums_set:
                    return False
        
        return True
    
    def add_element(S, sums_set, x):
        """Add x to S and update sums_set with all new triple sums."""
        n = len(S)
        S.append(x)
        sums_set.add(3 * x)
        for i in range(n):
            y = S[i]
            sums_set.add(2 * x + y)
            for j in range(i, n):
                z = S[j]
                sums_set.add(x + y + z)
    
    def greedy_with_order(candidates, max_elements=80):
        """Run greedy construction with a given candidate order."""
        S = []
        sums_set = set()
        
        for x in candidates:
            if len(S) >= max_elements:
                break
            if time.time() > deadline:
                break
            
            if is_valid_addition(S, x, sums_set):
                add_element(S, sums_set, x)
        
        return S
    
    # Strategy 1: Exponential spacing (powers of 2 or similar)
    # This tends to create unique sums because each element is much larger than the sum of smaller ones
    exp_candidates = []
    val = 1
    while val <= 4096:
        exp_candidates.append(val)
        val = int(val * 1.5) + 1  # 1.5x growth
    # Add some extra candidates near the top
    for v in range(2048, 4097, 100):
        if v not in exp_candidates:
            exp_candidates.append(v)
    exp_candidates.sort()
    
    result = greedy_with_order(exp_candidates)
    if len(result) > best_len:
        best_len = len(result)
        best_result = result
    
    # Strategy 2: Start with well-spaced base, then fill gaps
    # Use a base set with large gaps, then try to add elements in between
    def construct_with_base(base_step=64):
        """Build a set with base spacing, then try to fill in."""
        # Start with evenly spaced elements
        base_elements = list(range(1, 4097, base_step))
        S = []
        sums_set = set()
        
        # First pass: add base elements
        for x in base_elements:
            if len(S) >= 80:
                break
            if time.time() > deadline:
                break
            if is_valid_addition(S, x, sums_set):
                add_element(S, sums_set, x)
        
        if len(S) >= 80:
            return S
        
        # Second pass: try to fill in gaps
        S_sorted = sorted(S)
        candidates_to_try = []
        for i in range(len(S_sorted) - 1):
            lo = S_sorted[i]
            hi = S_sorted[i + 1]
            # Try a few candidates between lo and hi
            for frac in [0.25, 0.5, 0.75]:
                mid = int(lo + frac * (hi - lo))
                if 1 <= mid <= 4096 and mid not in S:
                    candidates_to_try.append(mid)
        
        # Also try small values not yet used
        for v in range(1, min(200, S_sorted[0] if S_sorted else 1)):
            if v not in S:
                candidates_to_try.append(v)
        
        # Shuffle candidates for variety
        random.Random(42).shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            if len(S) >= 80:
                break
            if time.time() > deadline:
                break
            if is_valid_addition(S, x, sums_set):
                add_element(S, sums_set, x)
        
        return S
    
    for step in [32, 64, 128, 256]:
        if time.time() > deadline - 2:
            break
        result = construct_with_base(step)
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    # Strategy 3: Local Search / Polish
    # We take the best result found so far and try to improve it via swaps and additions.
    if best_result:
        # Copy the best result to work on
        current_S = list(best_result)
        
        # Reconstruct sums_set for current_S
        # This is O(k^3) which is fine for k < 40
        current_sums = set()
        k = len(current_S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    current_sums.add(current_S[i] + current_S[j] + current_S[l])
        
        used_values = set(current_S)
        
        def remove_element(S, sums_set, x):
            """Remove x from S and sums_set. Returns new sums_set (or modifies in place if careful).
            Since sums_set is a set, we can't easily remove specific sums without knowing which ones involved x.
            It's safer to rebuild sums_set if we do many swaps, or track counts.
            For simplicity and speed, if k is small, rebuilding is O(k^3).
            """
            pass # We will use a helper to rebuild or update carefully.
            
        def rebuild_sums(S):
            s = set()
            k = len(S)
            for i in range(k):
                for j in range(i, k):
                    for l in range(j, k):
                        s.add(S[i] + S[j] + S[l])
            return s

        # Local Search Loop
        # We alternate between trying to add new elements and swapping existing ones.
        no_improvement_count = 0
        max_no_improvement = 2000 # Cap to avoid infinite loops if stuck
        
        while time.time() < deadline - 1 and no_improvement_count < max_no_improvement:
            improved = False
            
            # 1. Try to add a random element
            # Generate a batch of random candidates to try adding
            rng = random.Random(int(time.time() * 1000) % 10000)
            candidates_to_add = rng.sample([v for v in range(1, 4097) if v not in used_values], min(100, len(used_values) + 100))
            
            for x in candidates_to_add:
                if is_valid_addition(current_S, x, current_sums):
                    add_element(current_S, current_sums, x)
                    used_values.add(x)
                    improved = True
                    no_improvement_count = 0
                    break
                if time.time() > deadline - 1:
                    break
            
            if improved:
                continue
                
            # 2. Try to swap an element
            # Pick a random element to remove
            if current_S:
                idx_to_remove = rng.randint(0, len(current_S) - 1)
                x_to_remove = current_S[idx_to_remove]
                
                # Try to find a replacement
                # We need to check if S \ {x} U {y} is valid.
                # This is equivalent to checking if y can be added to S \ {x}.
                # We don't have sums for S \ {x} directly.
                # We can rebuild sums for S \ {x} (O(k^3)) or just test validity against the full set minus x's contributions?
                # Simpler: Just rebuild sums for the potential new set if a candidate y looks promising?
                # Or, check validity of y against S, but ignoring sums that involve x?
                
                # Let's try a few random y's
                candidates_to_swap = rng.sample([v for v in range(1, 4097) if v not in used_values], min(50, 4096))
                
                for y in candidates_to_swap:
                    # Check if y is valid in S \ {x}
                    # We can do this by checking all triple sums in S \ {x} U {y}
                    # This is O(k^3) per check. If k=30, 30^3 = 27000 ops. 50 checks = 1.3M ops.
                    # This is fast enough for a few iterations per second.
                    
                    # Optimization: Pre-calculate sums of S \ {x}?
                    # Let's just do the full check for simplicity first.
                    
                    S_new = [v for i, v in enumerate(current_S) if i != idx_to_remove]
                    # Check if y can be added to S_new
                    # We can reuse is_valid_addition if we had sums_set for S_new.
                    # But we don't.
                    # Let's write a helper or just inline the check.
                    
                    valid = True
                    # Sums in S_new are all in current_sums, but some might be "dirty" (involved x).
                    # Actually, we just need to ensure no collision.
                    # Collisions can happen between:
                    # 1. Sums in S_new (already unique among themselves)
                    # 2. New sums involving y: 3y, 2y+s, y+s1+s2
                    # 3. Collision between New sums and Sums in S_new.
                    
                    # We can check if any new sum is in current_sums.
                    # BUT, current_sums contains sums involving x.
                    # If a new sum equals a sum that involved x, it's NOT a collision because that old sum is gone.
                    # So we need to know which sums in current_sums do NOT involve x.
                    
                    # This is getting complicated to track dynamically without counts.
                    # Given k is small (< 40), rebuilding sums_set for S_new is O(k^3) ~ 64000 ops.
                    # This is very fast in Python if optimized or even not.
                    # Let's rebuild sums for S_new.
                    
                    sums_new = rebuild_sums(S_new)
                    
                    # Now check if y can be added to S_new using sums_new
                    if is_valid_addition(S_new, y, sums_new):
                        # Perform swap
                        current_S.pop(idx_to_remove)
                        add_element(current_S, sums_new, y)
                        current_sums = sums_new
                        used_values.remove(x_to_remove)
                        used_values.add(y)
                        improved = True
                        no_improvement_count = 0
                        break
                    if time.time() > deadline - 1:
                        break
                
                if not improved:
                    no_improvement_count += 1
            
            if len(current_S) > best_len:
                best_len = len(current_S)
                best_result = list(current_S)
        
        # Final check
        if len(current_S) > best_len:
            best_len = len(current_S)
            best_result = list(current_S)

    return best_result if best_result else [1, 2, 4]
# EVOLVE-BLOCK-END
