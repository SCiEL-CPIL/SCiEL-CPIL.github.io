import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    n = 128
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) within 0..n-1
    progs = []
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                progs.append((a, b, c, e))
    
    if not progs:
        return list(range(n))
        
    # Convert to numpy array for vectorized operations
    # Shape: (num_progs, 4)
    prog_indices = np.array(progs, dtype=np.int32)
    num_progs = len(prog_indices)
    
    # We want to minimize the number of distinct colors used, subject to constraints.
    # Constraint: Each progression must have >= 3 distinct colors.
    # Violation: A progression has < 3 distinct colors (i.e., 1 or 2 distinct colors).
    
    # We will try to find a coloring with K colors.
    # Strategy: Start with random 3-coloring, optimize violations.
    # If stuck with violations, try 4 colors, etc.
    # However, the objective is to MINIMIZE colors used.
    # So we first try to find ANY valid coloring with 3 colors.
    # If found, return it.
    # If not, try 4 colors.
    
    # Helper to count distinct colors in a vector of 4 values
    # We can compute violations vectorially.
    # For a row [c1, c2, c3, c4]:
    # Distinct count < 3 means:
    # 1. All 4 same: c1==c2==c3==c4
    # 2. 3 same, 1 diff: (c1==c2==c3 != c4) or (c1==c2==c4 != c3) or (c1==c3==c4 != c2) or (c2==c3==c4 != c1)
    # 3. 2 pairs: (c1==c2 != c3==c4) or (c1==c3 != c2==c4) or (c1==c4 != c2==c3)
    # Note: The condition ">= 3 distinct" is equivalent to NOT (<= 2 distinct).
    # It is easier to check if distinct count >= 3.
    # Alternatively, check if distinct count < 3.
    
    # Let's use a bit-packing trick or simple comparison.
    # Since colors are small integers, we can use sorting or set logic, but vectorized set is hard.
    # Comparison logic:
    # Let A = prog_indices[:, 0], B = prog_indices[:, 1], C = prog_indices[:, 2], D = prog_indices[:, 3]
    # Values: v0, v1, v2, v3
    # Distinct < 3 if:
    # (v0==v1 and v0==v2 and v0==v3) OR
    # (v0==v1 and v0==v2 and v2!=v3) OR # 3 same
    # ... this is getting complex.
    
    # Simpler: Compute number of unique values.
    # We can use a trick: 
    # If we map colors to bits, we can't easily count distinct.
    # Let's just check the condition "len(set) < 3" using broadcasting.
    # For each progression, we have 4 values.
    # We can create a matrix of differences? No.
    
    # Efficient check for "distinct count < 3":
    # It fails if all pairs are not distinct? No.
    # It fails if the number of distinct values is 1 or 2.
    # 
    # Let's use the property: 
    # distinct_count < 3  <=>  (v0==v1 and v0==v2 and v0==v3) 
    #                         or (v0==v1 and v0==v2 and v2!=v3) 
    #                         or (v0==v1 and v0==v3 and v1!=v2) # wait, v0==v1==v3, v2 diff
    #                         ...
    # 
    # Actually, it's easier to check the complement: distinct_count >= 3.
    # This is true if there are at least 3 values that are pairwise distinct? No, just 3 distinct values in the set.
    
    # Let's try a different approach: 
    # For each progression, sort the 4 values? Vectorized sort is possible but maybe slow.
    # 
    # Let's use the following identity for 4 values a,b,c,d:
    # They have < 3 distinct values if and only if:
    # (a==b and a==c and a==d) OR
    # (a==b and a==c and c!=d) OR
    # (a==b and a==d and b!=c) OR
    # (a==c and a==d and c!=b) OR
    # (b==c and b==d and c!=a) OR
    # (a==b and c==d and b!=c) OR
    # (a==c and b==d and a!=b) OR
    # (a==d and b==c and a!=b)
    
    # This covers:
    # 4 same: covered by first case.
    # 3 same: covered by cases 2-5.
    # 2 pairs: covered by cases 6-8.
    # 1 pair, 2 distinct: NOT covered. This is valid (3 distinct colors).
    # All distinct: NOT covered. This is valid (4 distinct colors).
    
    # So, Violation = 
    #   (v0==v1 & v0==v2 & v0==v3) |
    #   (v0==v1 & v0==v2 & v2!=v3) |
    #   (v0==v1 & v0==v3 & v1!=v2) |
    #   (v0==v2 & v0==v3 & v2!=v1) |
    #   (v1==v2 & v1==v3 & v2!=v0) |
    #   (v0==v1 & v2==v3 & v0!=v2) |
    #   (v0==v2 & v1==v3 & v0!=v1) |
    #   (v0==v3 & v1==v2 & v0!=v1)
    
    def get_violations(colors):
        v0 = colors[prog_indices[:, 0]]
        v1 = colors[prog_indices[:, 1]]
        v2 = colors[prog_indices[:, 2]]
        v3 = colors[prog_indices[:, 3]]
        
        # 4 same
        mask_4_same = (v0 == v1) & (v0 == v2) & (v0 == v3)
        
        # 3 same
        mask_3_same = (
            ((v0 == v1) & (v0 == v2) & (v2 != v3)) |
            ((v0 == v1) & (v0 == v3) & (v1 != v2)) |
            ((v0 == v2) & (v0 == v3) & (v2 != v1)) |
            ((v1 == v2) & (v1 == v3) & (v2 != v0))
        )
        
        # 2 pairs
        mask_2_pairs = (
            ((v0 == v1) & (v2 == v3) & (v0 != v2)) |
            ((v0 == v2) & (v1 == v3) & (v0 != v1)) |
            ((v0 == v3) & (v1 == v2) & (v0 != v1))
        )
        
        violations = mask_4_same | mask_3_same | mask_2_pairs
        return np.sum(violations)

    def optimize(num_colors, initial_temp=1.0, cooling=0.999, max_iters=500000):
        # Initialize with random colors in [0, num_colors)
        colors = np.random.randint(0, num_colors, n)
        current_violations = get_violations(colors)
        best_colors = colors.copy()
        best_violations = current_violations
        temp = initial_temp
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if current_violations == 0:
                # Check if we can reduce colors used?
                # For now, if 0 violations, we have a valid coloring.
                # We want to minimize the number of labels used.
                # If we are trying 3 colors, and we find a valid one, we should check if it uses < 3 colors?
                # No, if it uses < 3 colors, it would have been found with fewer colors.
                # But the constraint is "at least 3 distinct colors in every progression".
                # If the whole array uses only 2 colors, then every progression uses <= 2 colors, so it's invalid.
                # So a valid 3-coloring must use at least 3 colors?
                # Actually, if a valid coloring uses only 2 colors globally, it's impossible because every progression would use <= 2 colors.
                # So any valid coloring must use >= 3 colors.
                # Thus, finding a valid coloring with 3 available colors implies it uses exactly 3 (or more, but max is 3).
                return best_colors, 0
                
            # Pick a random position and change its color
            pos = np.random.randint(n)
            old_color = colors[pos]
            new_color = np.random.randint(num_colors)
            if new_color == old_color:
                continue
                
            colors[pos] = new_color
            new_violations = get_violations(colors)
            
            delta = new_violations - current_violations
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
            else:
                colors[pos] = old_color
                
            temp *= cooling
            
        return best_colors, best_violations

    # Try 3 colors first
    best_3, viol_3 = optimize(3)
    if viol_3 == 0:
        # Verify and return
        # Ensure labels are in 0..127
        return best_3.tolist()
        
    # If 3 colors failed (still violations), try 4 colors
    best_4, viol_4 = optimize(4)
    if viol_4 == 0:
        return best_4.tolist()
        
    # If 4 colors failed, try 5
    best_5, viol_5 = optimize(5)
    if viol_5 == 0:
        return best_5.tolist()
        
    # Fallback: Greedy or Random restarts if SA failed
    # Just return the best found so far, even if invalid? 
    # The task says "Every nonconstant... must use AT LEAST three".
    # If we can't find a valid one, we should try harder or return a valid 4-coloring if we found one.
    # If we didn't find any valid one, we might return an invalid one, which scores 0.
    # Let's do a quick greedy check on the best_4 if it's invalid? No, greedy is slow.
    
    # If we are here, we likely timed out or got stuck.
    # Return the best coloring found with fewest violations, hoping it's valid or close.
    # But we need to return a valid one to get > 0 score.
    # Let's try a few more restarts with 4 colors if we have time.
    
    if time.time() < deadline - 5:
        for _ in range(10):
            best_4, viol_4 = optimize(4, initial_temp=2.0, cooling=0.9995)
            if viol_4 == 0:
                return best_4.tolist()
                
    # Last resort: Return the 4-color attempt even if invalid? 
    # Better to return a known valid structure if we have one? 
    # We don't have a hardcoded one.
    # If we must return something, return best_4.
    return best_4.tolist()
