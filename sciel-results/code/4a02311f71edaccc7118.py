"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    colors = []
    n = 128
    
    for x in range(1, n + 1):
        if time.time() > deadline:
            # Return best partial result padded to length n
            # This should rarely happen with the optimized check
            break
        
        found = False
        for color in range(128):
            # Check constraint 1: every 4-term AP must use at least 3 colors
            # For position x, check all APs ending at x: (x-3d, x-2d, x-d, x)
            # Need at least 3 distinct colors among these 4 positions
            valid = True
            
            # Check 4-term APs: x-3d, x-2d, x-d, x
            # d ranges from 1 to (x-1)//3
            max_d_4 = (x - 1) // 3
            for d in range(1, max_d_4 + 1):
                idx3 = x - 1 - 3*d  # 0-indexed
                idx2 = x - 1 - 2*d
                idx1 = x - 1 - d
                # These must all be valid indices (>= 0)
                if idx3 < 0 or idx2 < 0 or idx1 < 0:
                    continue
                c3 = colors[idx3]
                c2 = colors[idx2]
                c1 = colors[idx1]
                # Need at least 3 distinct colors among {c3, c2, c1, color}
                if len({c3, c2, c1, color}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check constraint 2: no 3-term AP can be monochromatic
            # For position x, check all 3-term APs ending at x: (x-2d, x-d, x)
            # Need at least 2 distinct colors among these 3 positions
            max_d_3 = (x - 1) // 2
            for d in range(1, max_d_3 + 1):
                idx2 = x - 1 - 2*d
                idx1 = x - 1 - d
                if idx2 < 0 or idx1 < 0:
                    continue
                c2 = colors[idx2]
                c1 = colors[idx1]
                # Need at least 2 distinct colors among {c2, c1, color}
                if c2 == c1 == color:
                    valid = False
                    break
            
            if valid:
                colors.append(color)
                found = True
                break
        
        if not found:
            # No valid color found - this shouldn't happen with 128 colors
            # Pad with 0 to maintain length
            colors.extend([0] * (n - len(colors)))
            break
    
    # Ensure we have exactly n colors
    while len(colors) < n:
        colors.append(0)
    
    return colors[:n]
# EVOLVE-BLOCK-END
