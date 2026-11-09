"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import product

# EVOLVE-BLOCK-START
def _validate(coloring):
    """Check if coloring satisfies the 4-AP constraint."""
    n = len(coloring)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            # 4-term AP: a, a+d, a+2d, a+3d
            if a + 3*d >= n:
                continue
            c0, c1, c2, c3 = coloring[a], coloring[a+d], coloring[a+2*d], coloring[a+3*d]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _count_colors(coloring):
    """Count distinct colors used."""
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n = 128
    
    # Strategy: Try to find colorings with increasing number of colors
    # Start with 2 colors (likely infeasible), then 3, 4, etc.
    
    # First, try a simple greedy with limited colors
    # We'll use colors 0, 1, 2, ... and try to minimize distinct colors
    
    best_coloring = None
    best_num_colors = 128
    
    # Try different color counts from low to high
    for num_colors in range(2, 8):  # Try 2 to 7 colors
        if time.time() > deadline:
            break
        
        # Try multiple random restarts with this color count
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Greedy assignment with constraint checking
            coloring = [-1] * n
            coloring[0] = 0
            
            success = True
            for pos in range(1, n):
                if time.time() > deadline:
                    success = False
                    break
                
                # Try each color in order of least usage first
                color_usage = [0] * num_colors
                for c in coloring:
                    if c >= 0:
                        color_usage[c] += 1
                
                # Sort colors by usage (least used first)
                colors_to_try = sorted(range(num_colors), key=lambda c: color_usage[c])
                
                placed = False
                for c in colors_to_try:
                    # Check if placing color c at pos violates any 4-AP constraint
                    valid = True
                    
                    # Check all 4-term APs that include position pos
                    for d in range(1, n):
                        # pos could be at different positions in the AP
                        # Case 1: pos is first term: pos, pos+d, pos+2d, pos+3d
                        if pos + 3*d < n:
                            if coloring[pos+d] >= 0 and coloring[pos+2*d] >= 0 and coloring[pos+3*d] >= 0:
                                if len({c, coloring[pos+d], coloring[pos+2*d], coloring[pos+3*d]}) < 3:
                                    valid = False
                                    break
                        
                        # Case 2: pos is second term: pos-d, pos, pos+d, pos+2d
                        if pos - d >= 0 and pos + 2*d < n:
                            if coloring[pos-d] >= 0 and coloring[pos+d] >= 0 and coloring[pos+2*d] >= 0:
                                if len({coloring[pos-d], c, coloring[pos+d], coloring[pos+2*d]}) < 3:
                                    valid = False
                                    break
                        
                        # Case 3: pos is third term: pos-2d, pos-d, pos, pos+d
                        if pos - 2*d >= 0 and pos + d < n:
                            if coloring[pos-2*d] >= 0 and coloring[pos-d] >= 0 and coloring[pos+d] >= 0:
                                if len({coloring[pos-2*d], coloring[pos-d], c, coloring[pos+d]}) < 3:
                                    valid = False
                                    break
                        
                        # Case 4: pos is fourth term: pos-3d, pos-2d, pos-d, pos
                        if pos - 3*d >= 0:
                            if coloring[pos-3*d] >= 0 and coloring[pos-2*d] >= 0 and coloring[pos-d] >= 0:
                                if len({coloring[pos-3*d], coloring[pos-2*d], coloring[pos-d], c}) < 3:
                                    valid = False
                                    break
                    
                    if valid:
                        coloring[pos] = c
                        placed = True
                        break
                
                if not placed:
                    success = False
                    break
            
            if success and _validate(coloring):
                actual_colors = _count_colors(coloring)
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_coloring = coloring[:]
                    found = True
                    # If we found with this num_colors, we can stop trying higher
                    if actual_colors == num_colors:
                        break
        
        if found and best_num_colors <= num_colors:
            break
    
    # If we didn't find a good solution, fall back to the greedy approach
    if best_coloring is None or best_num_colors > 10:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color)
                    break
        best_coloring = colors
    
    return best_coloring
# EVOLVE-BLOCK-END
