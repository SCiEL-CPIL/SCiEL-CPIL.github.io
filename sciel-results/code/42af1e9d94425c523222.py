"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _check_valid(colors):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            prog = {colors[a], colors[b], colors[c], colors[e]}
            if len(prog) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_construct(max_colors=4):
    """Greedy construction trying to use at most max_colors colors."""
    colors = [0] * 128
    for x in range(128):
        # Try each color from 0 to max_colors-1
        found = False
        for color in range(max_colors):
            colors[x] = color
            # Check all 4-term progressions ending at or including position x
            valid = True
            # Progressions where x is the last element: a, a+d, a+2d, x where x=a+3d
            for d in range(1, (x) // 3 + 1):
                a = x - 3*d
                if a < 0:
                    break
                prog = {colors[a], colors[a+d], colors[a+2*d], colors[x]}
                if len(prog) < 3:
                    valid = False
                    break
            if valid:
                # Also check progressions where x is not the last element
                # x, x+d, x+2d, x+3d - but these aren't fully assigned yet
                # We only need to check progressions that are complete
                found = True
                break
        if not found:
            return None
    return colors

def _local_search(colors, deadline, max_colors=4):
    """Local search to minimize number of colors used."""
    colors = list(colors)
    best = _count_colors(colors)
    
    # Try to reduce colors by remapping
    # First, get the color mapping
    unique_colors = sorted(set(colors))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    remapped = [color_map[c] for c in colors]
    
    # Try to merge colors
    while best > 1:
        improved = False
        for c1 in range(best):
            for c2 in range(c1 + 1, best):
                # Try merging c2 into c1
                new_colors = [c1 if c == c2 else c for c in remapped]
                if _check_valid(new_colors):
                    remapped = new_colors
                    best -= 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return remapped

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10s safety margin
    
    # Try greedy construction with increasing number of colors
    for max_colors in range(3, 128):
        if time.time() > deadline:
            break
        colors = _greedy_construct(max_colors)
        if colors is not None and _check_valid(colors):
            # Try to minimize colors used
            if time.time() < deadline - 5:
                optimized = _local_search(colors, deadline)
                if _check_valid(optimized):
                    return optimized
            return colors
    
    # Fallback: use original greedy approach if time remains
    if time.time() < deadline - 10:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color)
                    break
        return colors
    
    # Emergency fallback
    return list(range(128))
# EVOLVE-BLOCK-END
