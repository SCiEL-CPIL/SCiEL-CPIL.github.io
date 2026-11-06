import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Precompute all 4-term progressions (a, b, c, d) with a < b < c < d
    # where b-a = c-b = d-c = step > 0
    # We need: for every such progression, len({color[a], color[b], color[c], color[d]}) >= 3
    # Equivalently: NOT (at most 2 distinct colors)
    
    # Let's index 0..127 for positions 1..128
    
    # Greedy with backtracking / local search approach
    # Since n=128 is small, we can try to find a coloring with few colors
    
    # First, let's try a greedy approach with limited colors
    # The problem asks to minimize the number of labels actually used
    
    # Let's try with 3 colors first, then 4, etc.
    # But we need to be fast.
    
    # Actually, let's use a smarter approach:
    # 1. Try greedy with colors 0, 1, 2, ... in order
    # 2. If stuck, backtrack or use local search
    
    # For speed, let's precompute the constraints
    # For each position i and each candidate color c, 
    # check if it violates any constraint with already-colored positions
    
    # Let's use a different strategy: 
    # Randomized local search starting from a random valid-looking assignment
    
    # Actually, for n=128, greedy should work if we allow enough colors.
    # The key insight: we want to MINIMIZE colors used.
    
    # Let's try: assign colors greedily, but only from a small set {0, 1, 2, ...}
    # and see how many colors we need.
    
    # But the greedy might fail. Let's use backtracking with pruning.
    
    # Given the time constraint, let's do:
    # 1. Try greedy with colors 0, 1, 2 (3 colors)
    # 2. If fails, try 0, 1, 2, 3 (4 colors)
    # 3. If fails, try more colors
    # 4. If all fail, use a fallback valid coloring with many colors
    
    # Precompute progressions for fast constraint checking
    # For each position i, list of (i-d, i-2d, i-3d) for valid d
    # and (i-2d, i-d, i+d) for future constraints
    
    # Actually, let's just do a simple greedy and hope it works with few colors.
    
    colors = [0] * 128
    
    for max_colors in range(1, 129):
        if time.time() - start_time > deadline - 5.0:
            break
            
        # Try greedy with colors 0..max_colors-1
        trial = [0] * 128
        valid = True
        
        for x in range(128):
            found = False
            for c in range(max_colors):
                ok = True
                
                # Check 4-term progressions ending at x
                # x - 3d >= 0
                for d in range(1, x // 3 + 1):
                    c1 = trial[x - d]
                    c2 = trial[x - 2*d]
                    c3 = trial[x - 3*d]
                    if len({c, c1, c2, c3}) < 3:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # Check progressions where x is the 2nd or 3rd element
                # x is 2nd: a, x, x+d, x+2d where a = x-d >= 0
                # x is 3rd: a, a+d, x, x+d where a = x-2d >= 0
                # For now, just check if it creates a dead end
                # Simplified: check if there's a progression where 3 of 4 are fixed
                # and the 4th is forced to violate
                
                # Actually, let's just do the basic check for now
                # and rely on the greedy working with enough colors
                
                trial[x] = c
                found = True
                break
            
            if not found:
                valid = False
                break
        
        if valid:
            # Count distinct colors used
            distinct = len(set(trial))
            if distinct <= max_colors:
                return trial
    
    # Fallback: use a simple valid coloring with many colors
    # Assign color = i % 128 (all distinct)
    # This trivially satisfies the constraint since all 4 elements are distinct
    return [i for i in range(128)]
