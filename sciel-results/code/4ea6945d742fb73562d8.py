"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Try to find coloring with minimal number of colors
    # Start with 3 colors (theoretical minimum for 4-term progressions)
    for num_colors in range(3, 10):
        if time.time() > deadline:
            break
        result = try_coloring(num_colors, deadline)
        if result is not None:
            return result
    
    # Fallback: use the original greedy approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 
                    for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) 
                        for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors

def is_valid_coloring(colors, n=128):
    """Check if coloring is valid: every nonconstant 4-term progression uses >=3 colors"""
    for a in range(n):
        for d in range(1, (n-1-a)//3 + 1):
            if d == 0:
                continue
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            # Check if this is a nonconstant progression (d != 0)
            colors_in_prog = {colors[a], colors[b], colors[c], colors[e]}
            if len(colors_in_prog) < 3:
                return False
    return True

def try_coloring(num_colors, deadline):
    """Try to find a valid coloring using at most num_colors colors"""
    colors = [-1] * 128
    palette = list(range(num_colors))
    
    # Use randomized backtracking with time limit
    random.seed(42)  # Deterministic for reproducibility
    
    def backtrack(pos):
        if time.time() > deadline:
            return None
        if pos == 128:
            return colors[:]
        
        # Try each color in random order for diversity
        shuffled = palette[:]
        random.shuffle(shuffled)
        
        for color in shuffled:
            colors[pos] = color
            if is_partial_valid(colors, pos):
                result = backtrack(pos + 1)
                if result is not None:
                    return result
        colors[pos] = -1
        return None
    
    def is_partial_valid(colors, pos):
        """Check if partial coloring up to pos is valid"""
        # Check all 4-term progressions that end at or before pos
        for a in range(pos + 1):
            max_d = (pos - a) // 3
            for d in range(1, max_d + 1):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                if e > pos:
                    break
                # Only check if all positions are assigned
                if colors[a] >= 0 and colors[b] >= 0 and colors[c] >= 0 and colors[e] >= 0:
                    colors_in_prog = {colors[a], colors[b], colors[c], colors[e]}
                    if len(colors_in_prog) < 3:
                        return False
        return True
    
    result = backtrack(0)
    if result is not None and is_valid_coloring(result):
        return result
    return None

# EVOLVE-BLOCK-END
