import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Helper functions
    def points_valid(pts):
        if len(pts) != 20:
            return False
        seen = set()
        for x, y in pts:
            if not (-100 <= x <= 100 and -100 <= y <= 100):
                return False
            if (x, y) in seen:
                return False
            seen.add((x, y))
        
        # Check no 4 collinear
        # For every pair, count how many points are on the line
        # Optimization: just check all subsets of 4? No, too slow.
        # Check all triples? If a triple is collinear, check if any 4th point is on the line.
        # Actually, simpler: for each line defined by 2 points, count points.
        # But there are $\binom{20}{2} = 190$ pairs.
        # For each pair, calculate line equation or just use cross product with all other points.
        # If count >= 4, invalid.
        
        for i in range(20):
            x1, y1 = pts[i]
            for j in range(i + 1, 20):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                count = 2
                for k in range(20):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    # Cross product (x3-x1)*dy - (y3-y1)*dx == 0
                    if (x3 - x1) * dy == (y3 - y1) * dx:
                        count += 1
                if count >= 4:
                    return False
        return True

    def count_triples(pts):
        count = 0
        for i in range(20):
            x1, y1 = pts[i]
            for j in range(i + 1, 20):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, 20):
                    x3, y3 = pts[k]
                    if (x3 - x1) * dy == (y3 - y1) * dx:
                        count += 1
        return count

    def max_independent_set(pts):
        # Precompute collinear triples as bitmasks
        # Bitmask of 20 bits.
        # If a triple is collinear, we cannot include all 3 in the independent set.
        # We want max subset with NO collinear triple.
        
        triples = []
        for i in range(20):
            x1, y1 = pts[i]
            for j in range(i + 1, 20):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, 20):
                    x3, y3 = pts[k]
                    if (x3 - x1) * dy == (y3 - y1) * dx:
                        mask = (1 << i) | (1 << j) | (1 << k)
                        triples.append(mask)
        
        # Branch and bound
        # State: available_mask (bits not yet decided/removed), current_set_mask
        # We want to maximize popcount(current_set_mask)
        # Constraint: current_set_mask must not contain any triple from `triples`
        
        # Optimization: If current_set_mask already contains a triple, it's invalid.
        # We can prune if (current_set_mask & triple_mask) == triple_mask for any triple.
        
        # Better: Precompute for each bit, which triples it belongs to?
        # Or just check validity when adding a bit.
        
        # Let's use a recursive function.
        # To speed up, sort points or use heuristic ordering?
        # With N=20, simple B&B is fast enough if we prune well.
        
        import functools
        
        @functools.lru_cache(maxsize=None)
        def solve(available_mask, current_mask):
            # Check if current_mask is valid
            # If not, return -inf
            # Actually, we only add bits if they don't form a triple with existing bits.
            # So current_mask is always valid.
            
            # Pruning: Upper bound
            # Max possible size = popcount(current_mask) + popcount(available_mask)
            # If this is <= best_found, prune.
            # We need to pass best_found or use global.
            
            # Let's use a global variable for best found in this call.
            # Since lru_cache doesn't support mutable state easily, we'll do iterative or pass best.
            # Or just return the max size from this state.
            
            if available_mask == 0:
                return bin(current_mask).count('1')
            
            # Heuristic: pick a bit from available_mask
            # Try adding bit i
            # If adding bit i creates a triple with current_mask, skip.
            
            best_val = 0
            # We can also choose to NOT add any more bits (stop here)
            # But we want max, so we should try to add.
            
            # To prune, we can calculate an upper bound.
            # A simple upper bound is popcount(available_mask) + popcount(current_mask)
            # If this <= best_global, we could prune, but we don't have best_global in cache.
            
            # Let's just recurse.
            # Pick first set bit in available_mask
            # Actually, to be faster, we can iterate through bits.
            
            # Optimization: If we pick a bit, we must remove it from available.
            # Also, if adding bit i creates a triple with two bits in current_mask, we can't add it.
            
            # Check validity of adding bit i
            # Triple (i, j, k) is invalid if j, k in current_mask.
            # Precompute for each i, the set of pairs (j,k) in current_mask that form a triple with i?
            # That depends on current_mask.
            
            # Let's just check all triples involving i.
            
            # Iterate through available bits
            # To reduce branching, we can use symmetry breaking or heuristic.
            # For N=20, even without strong pruning, it's likely fast.
            
            # Let's try a simple approach:
            # Pick a bit i from available.
            # Option 1: Include i.
            #   Check if any triple (i, j, k) has j,k in current_mask.
            #   If not, new_current = current_mask | (1<<i)
            #   new_available = available_mask & ~(1<<i)
            #   # Also, we can remove bits that form a triple with i and another bit in new_current?
            #   # No, we just ensure no triple is fully in the set.
            #   # If we include i, we cannot include any pair (j,k) such that (i,j,k) is a triple.
            #   # So we can prune available_mask by removing any bit that forms a triple with i and another bit in new_current?
            #   # Actually, if we include i, we must ensure that no other two bits in the final set form a triple with i.
            #   # So for any triple (i, j, k), we cannot have both j and k in the set.
            #   # This is a constraint on the remaining available bits.
            
            # This is getting complex. Let's stick to a simpler B&B.
            
            # Simple B&B:
            # State: (available_mask, current_mask)
            # If available is empty, return size.
            # Pick a bit i from available.
            # Try including i:
            #   If valid (no triple with i and 2 bits in current_mask), recurse(available \ {i}, current | {i})
            # Try excluding i:
            #   recurse(available \ {i}, current)
            # Return max.
            
            # To speed up, we can pick i such that it's in many triples?
            # Or just first bit.
            
            if available_mask == 0:
                return bin(current_mask).count('1')
            
            # Pick first set bit
            i = 0
            while not (available_mask & (1 << i)):
                i += 1
            
            bit_i = 1 << i
            
            # Option 1: Exclude i
            val_exclude = solve(available_mask & ~bit_i, current_mask)
            
            # Option 2: Include i
            # Check if adding i creates a triple
            valid = True
            for t in triples:
                if (t & bit_i) and (t & current_mask & ~bit_i).bit_count() == 2:
                    # t has i, and 2 other bits in current_mask
                    valid = False
                    break
            
            val_include = -1
            if valid:
                val_include = solve(available_mask & ~bit_i, current_mask | bit_i)
            
            return max(val_exclude, val_include)
        
        # Clear cache for new points
        solve.cache_clear()
        return solve((1 << 20) - 1, 0)

    # Initial points: Parabola
    best_points = [[x, x * x] for x in range(-10, 10)]
    best_score = 20
    
    # SA Search
    # Start with a perturbed grid
    # Grid 4x5
    grid_points = []
    for x in range(4):
        for y in range(5):
            grid_points.append([x * 10, y * 10])
    
    # Perturb to break 4-collinear
    # Randomly shift points
    current_points = [p[:] for p in grid_points]
    for p in current_points:
        p[0] += random.randint(-2, 2)
        p[1] += random.randint(-2, 2)
    
    # Ensure validity
    if not points_valid(current_points):
        # Fallback to random
        current_points = []
        while len(current_points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in current_points:
                current_points.append([x, y])
        if not points_valid(current_points):
            current_points = best_points[:]
    
    T = 10.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Propose move
        idx = random.randint(0, 19)
        old_point = current_points[idx][:]
        
        # New point
        # Local move
        if random.random() < 0.5:
            new_x = old_point[0] + random.randint(-5, 5)
            new_y = old_point[1] + random.randint(-5, 5)
        else:
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
        
        new_point = [new_x, new_y]
        
        # Check bounds
        if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
            continue
        
        # Check distinctness
        if any(new_point == p for p in current_points if p != old_point):
            continue
        
        # Swap
        current_points[idx] = new_point
        
        # Check validity
        if not points_valid(current_points):
            current_points[idx] = old_point
            T *= cooling
            continue
        
        # Evaluate
        # Use triple count as energy
        new_triples = count_triples(current_points)
        old_triples = count_triples(current_points) # This is inefficient, store old_triples
        
        # We didn't store old_triples. Let's restructure.
        
        T *= cooling

    # This loop is inefficient. Let's rewrite SA properly.
    
    return best_points

# Rewrite construct with proper SA
