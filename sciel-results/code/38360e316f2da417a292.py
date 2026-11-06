"""Coloring problem: minimize colors used subject to every nonconstant 4-AP using >=3 colors."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin within 90s limit
    n = 128
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)
    
    # Precompute all nonconstant 4-term progressions: (a, a+d, a+2d, a+3d) with d>0
    # positions 0..127
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a + d, a + 2*d, a + 3*d))
    
    n_aps = len(aps)
    
    # Convert to numpy arrays for vectorized operations
    ap_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # We'll try colorings with 2, 3, 4, 5 colors
    # For k colors, each position gets a value in 0..k-1
    # Constraint: for each AP (i,j,l,m), colors[i], colors[j], colors[l], colors[m] must have >=3 distinct values
    
    # Objective: minimize number of distinct colors used
    # But we need a valid coloring first
    
    # Strategy: 
    # 1. Try to find valid coloring with 3 colors (greedy + repair)
    # 2. If found, try to reduce to 2 colors (likely impossible but check)
    # 3. Use simulated annealing to minimize color count while maintaining validity
    
    # First, let's try a greedy approach with 3 colors
    # Then use SA to optimize
    
    best_colors = None
    best_num_colors = 128  # worst case
    
    # Try to find a valid 3-coloring first using the seed's greedy approach
    # but with randomization for diversity
    for attempt in range(20):
        if time.time() > deadline:
            break
        colors = [-1] * n
        valid = True
        for pos in range(n):
            # Try each color 0,1,2
            found = False
            for c in range(3):
                ok = True
                # Check all APs that include this position
                for idx in range(n_aps):
                    ap = ap_arr[idx]
                    if pos not in ap:
                        continue
                    # Check if assigning c to pos violates the >=3 distinct constraint
                    # given current assignments
                    vals = []
                    for p in ap:
                        if p == pos:
                            vals.append(c)
                        elif colors[p] >= 0:
                            vals.append(colors[p])
                        else:
                            vals.append(None)  # unassigned
                    # We need at least 3 distinct values among the 4 positions
                    # But unassigned positions can be anything, so we only need to check
                    # that the assigned ones don't force < 3 colors
                    assigned = [v for v in vals if v is not None]
                    if len(set(assigned)) >= 3:
                        continue  # already satisfied
                    # Check if it's possible to have >= 3 distinct colors
                    # The unassigned positions can provide new colors
                    num_unassigned = len(vals) - len(assigned)
                    # We need at least 3 - len(set(assigned)) new distinct colors from unassigned
                    # Each unassigned position can contribute at most 1 new color
                    if num_unassigned < 3 - len(set(assigned)):
                        ok = False
                        break
                if ok:
                    colors[pos] = c
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            # Verify
            num_distinct = len(set(colors))
            if num_distinct < best_num_colors:
                best_num_colors = num_distinct
                best_colors = colors[:]
            if best_num_colors == 3:
                break  # Can't do better than 3 (2 colors always fails for some AP)
    
    # Now use simulated annealing to try to improve
    # Start from best found or random
    if best_colors is None:
        # Random 3-coloring
        best_colors = [rng.randint(0, 2) for _ in range(n)]
        best_num_colors = 3
    
    # SA to minimize number of distinct colors used
    # Objective: count of distinct colors in the coloring
    # But we need to maintain validity
    
    def count_violations(colors):
        """Count number of APs that violate the >=3 distinct colors constraint."""
        colors_arr = np.array(colors, dtype=np.int8)
        violations = 0
        for idx in range(n_aps):
            ap = ap_arr[idx]
            vals = colors_arr[ap]
            if len(set(vals.tolist())) < 3:
                violations += 1
        return violations
    
    def count_distinct(colors):
        return len(set(colors))
    
    # Use SA: minimize (violations * big_penalty + distinct_colors)
    BIG_PENALTY = 1000
    
    def objective(colors):
        v = count_violations(colors)
        d = count_distinct(colors)
        return v * BIG_PENALTY + d
    
    # Simulated annealing
    current = best_colors[:]
    current_obj = objective(current)
    best = current[:]
    best_obj = current_obj
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    steps_per_temp = 50
    
    sa_start = time.time()
    sa_deadline = min(deadline, sa_start + 60)
    
    while T > T_min:
        if time.time() > sa_deadline:
            break
        for _ in range(steps_per_temp):
            if time.time() > sa_deadline:
                break
            # Pick a random position and change its color
            pos = rng.randint(0, n - 1)
            old_color = current[pos]
            new_color = rng.randint(0, 2)
            if new_color == old_color:
                continue
            current[pos] = new_color
            new_obj = objective(current)
            delta = new_obj - current_obj
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best = current[:]
            else:
                current[pos] = old_color
        T *= cooling
    
    # Use best found
    if best_obj < objective(best_colors):
        best_colors = best
    
    # Try to reduce colors: if using 3 colors, try to map to fewer
    # Check if 2 colors suffice (unlikely but check)
    # For 2 colors, an AP with all same color or 3 same 1 different fails
    # Actually with 2 colors, any AP has at most 2 distinct colors, so it ALWAYS fails
    # So minimum is 3 colors
    
    # Final verification and cleanup
    # Map colors to 0, 1, 2 (or whatever subset is used)
    unique_colors = sorted(set(best_colors))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in best_colors]
    
    # Ensure result is valid
    final_colors = np.array(result, dtype=np.int8)
    valid = True
    for idx in range(n_aps):
        ap = ap_arr[idx]
        vals = final_colors[ap]
        if len(set(vals.tolist())) < 3:
            valid = False
            break
    
    if not valid:
        # Fall back to greedy
        colors = []
        for x in range(1, 129):
            for color in range(3):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2 + 1))):
                    colors.append(color)
                    break
        return colors
    
    return result
# EVOLVE-BLOCK-END
