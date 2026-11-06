"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def _check_valid(colors):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            e = a + 3 * d
            if e >= n:
                break
            prog = {colors[a], colors[a + d], colors[a + 2 * d], colors[e]}
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
                a = x - 3 * d
                if a < 0:
                    break
                prog = {colors[a], colors[a + d], colors[a + 2 * d], colors[x]}
                if len(prog) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors

def _local_search(colors, deadline):
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

def _solve_with_backtracking(n, max_colors, deadline):
    """Solve using backtracking with constraint checking."""
    colors = [-1] * n
    
    def check_assignment(pos, color):
        """Check if assigning color to pos violates any fully-assigned progression."""
        # Check all progressions containing pos
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                e = a + 3 * d
                if e >= n:
                    break
                if pos not in (a, a + d, a + 2 * d, e):
                    continue
                vals = [colors[a], colors[a + d], colors[a + 2 * d], colors[e]]
                # Substitute the new color for pos
                if pos == a:
                    vals[0] = color
                elif pos == a + d:
                    vals[1] = color
                elif pos == a + 2 * d:
                    vals[2] = color
                else:
                    vals[3] = color
                if -1 in vals:
                    continue
                if len(set(vals)) < 3:
                    return False
        return True
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        
        for color in range(max_colors):
            if not check_assignment(pos, color):
                continue
            colors[pos] = color
            if backtrack(pos + 1):
                return True
            colors[pos] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    
    # Try backtracking with increasing number of colors
    for max_colors in range(3, 6):
        if time.time() > deadline - 15:
            break
        
        colors = _solve_with_backtracking(n, max_colors, deadline - 10)
        
        if colors is not None and _check_valid(colors):
            if time.time() < deadline - 5:
                optimized = _local_search(colors, deadline)
                if _check_valid(optimized):
                    return optimized
            return colors
    
    # Fallback: greedy construction
    for max_colors in range(3, 128):
        if time.time() > deadline - 5:
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
