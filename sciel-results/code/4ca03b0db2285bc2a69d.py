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
        found = False
        for color in range(max_colors):
            colors[x] = color
            valid = True
            for d in range(1, (x) // 3 + 1):
                a = x - 3*d
                if a < 0:
                    break
                prog = {colors[a], colors[a+d], colors[a+2*d], colors[x]}
                if len(prog) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors

def _local_search(colors, deadline, max_colors=4):
    """Local search to minimize number of colors used."""
    colors = list(colors)
    best = _count_colors(colors)
    
    unique_colors = sorted(set(colors))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    remapped = [color_map[c] for c in colors]
    
    while best > 1:
        improved = False
        for c1 in range(best):
            for c2 in range(c1 + 1, best):
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

def _backtracking_search(max_colors, deadline):
    """Backtracking search to find valid coloring with at most max_colors colors."""
    n = 128
    colors = [-1] * n
    
    def is_valid_partial(x, color):
        # Check all 4-term progressions that include position x and are fully assigned
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                if e >= n:
                    break
                # Check if x is one of the positions
                if x in (a, b, c, e):
                    # All positions must be assigned
                    if -1 in (colors[a], colors[b], colors[c], colors[e]):
                        continue
                    prog = {colors[a], colors[b], colors[c], colors[e]}
                    if len(prog) < 3:
                        return False
        return True
    
    def backtrack(x):
        if time.time() > deadline:
            return False
        if x == n:
            return True
        for color in range(max_colors):
            if is_valid_partial(x, color):
                colors[x] = color
                if backtrack(x + 1):
                    return True
                colors[x] = -1
        return False
    
    if backtrack(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    # Try backtracking with increasing number of colors
    for max_colors in range(3, 10):
        if time.time() > deadline - 10:
            break
        colors = _backtracking_search(max_colors, deadline - 5)
        if colors is not None and _check_valid(colors):
            # Try to minimize colors used
            if time.time() < deadline - 5:
                optimized = _local_search(colors, deadline)
                if _check_valid(optimized):
                    return optimized
            return colors
    
    # Try greedy construction as fallback
    for max_colors in range(3, 128):
        if time.time() > deadline:
            break
        colors = _greedy_construct(max_colors)
        if colors is not None and _check_valid(colors):
            if time.time() < deadline - 5:
                optimized = _local_search(colors, deadline)
                if _check_valid(optimized):
                    return optimized
            return colors
    
    # Emergency fallback
    return list(range(128))
# EVOLVE-BLOCK-END
