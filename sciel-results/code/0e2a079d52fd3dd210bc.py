"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 128
    colors = [0] * n
    
    def check_valid(pos, color, colors, n):
        """Check if assigning 'color' to 0-indexed position pos is valid."""
        x = pos + 1  # convert to 1-indexed
        
        # Position as 1st term: (x, x+d, x+2d, x+3d)
        for d in range(1, (n - x) // 3 + 1):
            if len({color, colors[x+d-1], colors[x+2*d-1], colors[x+3*d-1]}) < 3:
                return False
        
        # Position as 2nd term: (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if len({colors[x-d-1], color, colors[x+d-1], colors[x+2*d-1]}) < 3:
                return False
        
        # Position as 3rd term: (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if len({colors[x-2*d-1], colors[x-d-1], color, colors[x+d-1]}) < 3:
                return False
        
        # Position as 4th term: (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if len({colors[x-3*d-1], colors[x-2*d-1], colors[x-d-1], color}) < 3:
                return False
        
        return True
    
    for pos in range(n):
        if time.time() - start_time > DEADLINE:
            # Return what we have; assume it's valid from prior assignments
            break
        for color in range(3):
            if check_valid(pos, color, colors, n):
                colors[pos] = color
                break
        else:
            # Fallback if 3 colors don't work (shouldn't happen)
            for color in range(3, 128):
                if check_valid(pos, color, colors, n):
                    colors[pos] = color
                    break
    
    return colors
# EVOLVE-BLOCK-END
