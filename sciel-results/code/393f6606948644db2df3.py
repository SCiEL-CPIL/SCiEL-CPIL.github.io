"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    colors = [0] * 128
    # Greedy backtracking with look-ahead: at each position try colors in order,
    # but prefer colors that leave more options for the next position.
    # We use a simple heuristic: try the smallest valid color first (deterministic),
    # but if we detect a conflict ahead, we can backtrack.
    
    # For this problem, a simple greedy approach with the constraint check
    # should work. Let's add a deadline guard and ensure we return something valid.
    
    # Precompute: for position x (1-indexed), which (d) values are relevant
    # Four-term progression ending at x: (x-3d, x-2d, x-d, x) for d=1..(x-1)//3
    # Two-term "not all same" check: actually the constraint is about 4-term progressions.
    # The second condition in the original checks that for 2-step progressions,
    # not all three are the same color (which would make any 4-term extension trivially bad).
    # Actually re-reading: the second condition ensures that for pairs (x-2d, x-d, x),
    # not all are equal, which is necessary but not sufficient.
    
    # Let me just use the original logic with a deadline guard.
    colors_list = []
    for x in range(1, 129):
        if time.time() > deadline:
            # Return what we have padded with a safe color
            while len(colors_list) < 128:
                colors_list.append(0)
            return colors_list
        
        found = False
        for color in range(128):
            valid = True
            # Check all 4-term progressions ending at x
            for d in range(1, (x - 1) // 3 + 1):
                prev_colors = [colors_list[x - 1 - j * d] for j in (1, 2, 3)]
                distinct = len({color} | set(prev_colors))
                if distinct < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check the auxiliary constraint (from original)
            for d in range(1, (x - 1) // 2 + 1):
                if colors_list[x - 1 - d] == color and colors_list[x - 1 - 2 * d] == color:
                    valid = False
                    break
            if not valid:
                continue
            
            colors_list.append(color)
            found = True
            break
        
        if not found:
            # Should not happen with 128 colors, but safety
            colors_list.append(0)
    
    return colors_list
# EVOLVE-BLOCK-END
