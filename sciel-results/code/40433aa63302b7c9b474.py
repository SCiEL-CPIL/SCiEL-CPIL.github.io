import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Leave 10s margin
    
    n = 128
    best_colors = None
    best_count = n + 1
    
    # Try a few random orders of color preference to find a coloring with fewer colors
    # The greedy algorithm picks the smallest valid color.
    # If we randomize the "smallest" (i.e., shuffle the color list), we might find a solution with fewer distinct colors.
    
    for attempt in range(20): # Limit attempts to ensure speed
        if time.time() > deadline - 5.0:
            break
            
        colors = [-1] * n
        # Shuffle the order in which we try colors
        # We want to try colors in a specific order. 
        # Standard greedy: 0, 1, 2...
        # Randomized greedy: shuffle [0, 1, ..., 127] and pick the first valid one from this list.
        # But we want to minimize the *number* of distinct colors used.
        # A simple strategy: Try to use only the first K colors? 
        # Or just standard greedy often works well.
        
        # Let's stick to a slightly smarter greedy:
        # Assign colors sequentially. For position i, try colors 0..127.
        # To minimize distinct colors, we prefer reusing existing colors.
        # So, try colors that are already used in the prefix first?
        # Or just try 0, 1, 2... and see how many distinct ones end up used.
        
        # Let's try standard greedy (smallest index) first.
        color_order = list(range(128))
        if attempt > 0:
            random.shuffle(color_order)
        
        valid = True
        for i in range(n):
            found = False
            for c in color_order:
                is_ok = True
                # Check all 4-APs ending at i
                # i = j + 3d => j = i - 3d
                max_d = i // 3
                for d in range(1, max_d + 1):
                    j = i - 3 * d
                    # AP: j, j+d, j+2d, i
                    c1 = colors[j]
                    c2 = colors[j + d]
                    c3 = colors[j + 2 * d]
                    # c4 = c
                    # Check distinct count >= 3
                    if len({c1, c2, c3, c}) < 3:
                        is_ok = False
                        break
                if is_ok:
                    colors[i] = c
                    found = True
                    break
            if not found:
                valid = False
                break
        
        if valid:
            used = set(colors)
            if len(used) < best_count:
                best_count = len(used)
                best_colors = colors[:]
                # If we found a very good one (e.g., 3 colors), we can stop or continue
                if best_count <= 3:
                    # 3 is the theoretical minimum for non-constant APs? 
                    # Actually, can we do 2? No, 2 colors always allow a monochromatic or 2-color AP?
                    # Van der Waerden number W(2,4) is 76? No, W(2,4) is the smallest N such that any 2-coloring has a mono 4-AP.
                    # W(2,4) = 76. So for N=128, any 2-coloring has a mono 4-AP.
                    # A mono 4-AP has 1 distinct color. < 3.
                    # So 2 colors is impossible.
                    # 3 colors is the minimum possible.
                    # If we find 3, we can break.
                    break

    if best_colors is None:
        # Fallback: Should not happen, but just in case
        # Use a simple deterministic assignment that is known to work with many colors
        # e.g., color i = i % 128 (uses 128 colors, always valid? 
        # If i = j+3d, colors are i, i-d, i-2d, i-3d. 
        # If mod 128, they are distinct unless d is multiple of 128/4? 
        # With 128 colors, we can just use i.
        # i, i-d, i-2d, i-3d. 
        # Are they distinct? 
        # i - (i-d) = d. 
        # i - (i-2d) = 2d.
        # i - (i-3d) = 3d.
        # If d, 2d, 3d are not 0 mod 128, they are distinct.
        # Max d for 4-AP in 128 is 42. 
        # 3*42 = 126 != 0.
        # So using color = i is valid and uses 128 colors.
        best_colors = list(range(128))
        best_count = 128

    return best_colors

# EVOLVE-BLOCK-END
