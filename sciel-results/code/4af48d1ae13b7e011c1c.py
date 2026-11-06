import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # 5s margin
    
    selected = []
    sums = set()
    candidates = set(range(1, N + 1))
    
    while candidates:
        if time.time() > deadline:
            break
            
        # Find valid candidates: those that don't collide with current sums
        valid = []
        for x in candidates:
            # Check if x can be added
            # Sums to check: {x+a for a in selected} U {2*x}
            # If any of these are in sums, x is invalid
            is_valid = True
            for a in selected:
                if (x + a) in sums:
                    is_valid = False
                    break
            if is_valid and (2 * x) in sums:
                is_valid = False
            if is_valid:
                valid.append(x)
                
        if not valid:
            break
            
        # If only one valid, pick it
        if len(valid) == 1:
            x = valid[0]
        else:
            # Heuristic: pick x that blocks the most other valid candidates
            best_x = None
            max_blocked = -1
            
            for x in valid:
                # Calculate which other valid candidates would be blocked by adding x
                # A candidate y is blocked if adding y to selected U {x} causes a collision.
                # Since y is currently valid against selected, it only collides with the NEW sums from x.
                # New sums from x: S_new = {x+a for a in selected} U {2*x}
                # y is blocked if:
                # 1. 2*y is in S_new
                # 2. y+x is in S_new
                # 3. y+a is in S_new for some a in selected
                
                # Let's derive the conditions for y being blocked by x:
                # S_new elements are of form x+b (b in selected) or 2*x.
                
                # Case 1: 2*y in S_new
                # 2*y = 2*x => y=x (not a candidate)
                # 2*y = x+b => y = (x+b)/2. Must be integer.
                
                # Case 2: y+x in S_new
                # y+x = 2*x => y=x (not a candidate)
                # y+x = x+b => y=b. Since y is candidate, y not in selected. So no solution.
                
                # Case 3: y+a in S_new for some a in selected
                # y+a = 2*x => y = 2*x - a
                # y+a = x+b => y = x - a + b
                
                blocked_candidates = set()
                for a in selected:
                    # From y = 2*x - a
                    y_val = 2 * x - a
                    if 1 <= y_val <= N:
                        blocked_candidates.add(y_val)
                    
                    # From y = x - a + b
                    for b in selected:
                        y_val = x - a + b
                        if 1 <= y_val <= N:
                            blocked_candidates.add(y_val)
                    
                    # From y = (x + a) / 2
                    if (x + a) % 2 == 0:
                        y_val = (x + a) // 2
                        if 1 <= y_val <= N:
                            blocked_candidates.add(y_val)
                
                # Count how many of these blocked candidates are actually in 'valid'
                # We need to intersect blocked_candidates with the set of valid candidates
                # To make this faster, we can just check membership in a set of valid candidates
                # But 'valid' is a list. Let's convert to set once outside? 
                # Or just check `if y in valid_set`
                
                # Optimization: precompute valid_set
                # We are inside the loop for x. valid_set is constant for this step.
                # We'll create valid_set once per step.
                
                # For now, let's just count intersections.
                # Since valid is small-ish, we can just do:
                # count = sum(1 for y in blocked_candidates if y in valid_set)
                
                # We'll compute valid_set before the x-loop.
                
                # Placeholder for count logic, will implement in main loop below
                pass

            # Re-do the loop with valid_set
            valid_set = set(valid)
            best_x = None
            max_blocked = -1
            for x in valid:
                blocked_candidates = set()
                for a in selected:
                    y_val = 2 * x - a
                    if 1 <= y_val <= N:
                        blocked_candidates.add(y_val)
                    
                    for b in selected:
                        y_val = x - a + b
                        if 1 <= y_val <= N:
                            blocked_candidates.add(y_val)
                    
                    if (x + a) % 2 == 0:
                        y_val = (x + a) // 2
                        if 1 <= y_val <= N:
                            blocked_candidates.add(y_val)
                
                # Count how many blocked candidates are in valid_set
                # We only care about blocking OTHER valid candidates
                blocked_count = 0
                for y in blocked_candidates:
                    if y in valid_set and y != x:
                        blocked_count += 1
                
                if blocked_count > max_blocked:
                    max_blocked = blocked_count
                    best_x = x
            
            x = best_x
            
        # Add x to selected
        selected.append(x)
        # Update sums
        for a in selected[:-1]: # previous selected
            sums.add(x + a)
        sums.add(2 * x)
        
        # Remove x from candidates
        candidates.discard(x)
        
        # Optionally, remove newly blocked candidates from 'candidates' early to speed up next iteration?
        # The 'valid' check at the start of the next loop handles this, but removing them now
        # reduces the size of 'candidates' and 'valid' computation.
        # However, determining which are blocked is the same work as the heuristic.
        # We can just let the next iteration's 'valid' filter handle it.
        
    return selected
# EVOLVE-BLOCK-END
