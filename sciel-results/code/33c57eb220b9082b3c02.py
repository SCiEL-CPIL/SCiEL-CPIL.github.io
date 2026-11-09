<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def _check_progressions(colors, n):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            vals = {colors[a], colors[b], colors[c], colors[e]}
            if len(vals) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    n = 128
    best_colors = None
    best_num_colors = 128
    
    # Try to find coloring with k colors for k = 3, 4, 5, ...
    for k in range(3, 8):
        if time.time() > deadline:
            break
        
        # Backtracking search with k colors
        colors = [-1] * n
        found = False
        
        def backtrack(pos):
            nonlocal found
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            # Try each color
            for color in range(k):
                colors[pos] = color
                
                # Check all progressions ending at or involving pos
                valid = True
                # Progressions where pos is the last element: (pos-3d, pos-2d, pos-d, pos)
                for d in range(1, pos // 3 + 1):
                    a = pos - 3*d
                    if a < 0:
                        break
                    vals = {colors[a], colors[a+d], colors[a+2*d], colors[pos]}
                    if len(vals) < 3:
                        valid = False
                        break
                
                # Progressions where pos is the second element: (pos-d, pos, pos+d, pos+2d) - but pos+d, pos+2d not set yet
                # Actually we need to check all progressions that include pos and have all elements set
                # Let's check all progressions that include pos
                for d in range(1, n):
                    # pos could be at position a, a+d, a+2d, or a+3d in a progression
                    # Case 1: pos = a (first element)
                    a = pos
                    if a + 3*d < n:
                        vals = {colors[a], colors[a+d], colors[a+2*d], colors[a+3*d]}
                        if all(c >= 0 for c in vals) and len(vals) < 3:
                            valid = False
                            break
                    # Case 2: pos = a+d (second element)
                    a = pos - d
                    if 0 <= a and a + 3*d < n:
                        vals = {colors[a], colors[pos], colors[a+2*d], colors[a+3*d]}
                        if all(c >= 0 for c in vals) and len(vals) < 3:
                            valid = False
                            break
                    # Case 3: pos = a+2d (third element)
                    a = pos - 2*d
                    if 0 <= a and a + 3*d < n:
                        vals = {colors[a], colors[a+d], colors[pos], colors[a+3*d]}
                        if all(c >= 0 for c in vals) and len(vals) < 3:
                            valid = False
                            break
                    # Case 4: pos = a+3d (fourth element)
                    a = pos - 3*d
                    if 0 <= a:
                        vals = {colors[a], colors[a+d], colors[a+2*d], colors[pos]}
                        if all(c >= 0 for c in vals) and len(vals) < 3:
                            valid = False
                            break
                
                if valid:
                    if backtrack(pos + 1):
                        return True
                else:
                    colors[pos] = -1
                    continue
            
            colors[pos] = -1
            return False
        
        if backtrack(0):
            best_colors = colors[:]
            best_num_colors = k
            break
    
    if best_colors is None:
        # Fallback: use the greedy approach
        best_colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {best_colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3+1))
                    and all(any(best_colors[x-1-j*d] != color for j in (1,2)) 
                           for d in range(1, (x-1)//2+1))):
                    best_colors.append(color)
                    break
    
    return best_colors
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the backtracking approach above is too slow because checking all progressions at each step is expensive. Let me optimize by only checking progressions that are fully determined (all 4 positions have been assigned).

Let me reconsider the approach. I'll use a more efficient backtracking that only checks constraints when all 4 positions in a progression are assigned.

