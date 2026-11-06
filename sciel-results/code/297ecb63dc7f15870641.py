"""Rewritten with constraint-based backtracking search for minimal-color 4-AP coloring."""
import time
import random
from itertools import product

DEADLINE = 85.0  # Safety margin before 90s kill

def _validate(colors):
    """Check if coloring is valid: every nonconstant 4-term AP uses >= 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if a + 3 * d >= n:
                break
            idxs = [a, a + d, a + 2 * d, a + 3 * d]
            if len(set(colors[i] for i in idxs)) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _precompute_aps(n=128):
    """Precompute all nonconstant 4-term APs as index tuples."""
    aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if a + 3 * d < n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _backtrack_search(colors, pos, aps, color_count, start_color=0):
    """Backtracking search to extend partial coloring."""
    if time.time() > DEADLINE:
        return None
    
    if pos == len(colors):
        return colors[:]
    
    # Try each color
    for c in range(start_color, color_count):
        colors[pos] = c
        # Check all 4-term APs that include position pos
        valid = True
        for (a, b, cc, dd) in aps:
            if pos in (a, b, cc, dd):
                idxs = (a, b, cc, dd)
                vals = [colors[i] for i in idxs]
                if any(v is None for v in vals):
                    continue
                if len(set(vals)) < 3:
                    valid = False
                    break
        if valid:
            result = _backtrack_search(colors, pos + 1, aps, color_count, 0)
            if result is not None:
                return result
        colors[pos] = None
    
    return None

def construct():
    start_time = time.time()
    n = 128
    aps = _precompute_aps(n)
    
    # Try 3 colors first (minimum possible)
    for attempt in range(50):
        if time.time() > DEADLINE:
            break
        
        # Use randomization to escape local traps
        random.seed(attempt * 12345 + 67890)
        
        colors = [None] * n
        # Fix first few positions to break symmetry
        # Try different initial patterns
        init_patterns = [
            [0, 1, 2],
            [0, 1, 0],
            [0, 2, 1],
            [1, 0, 2],
            [0, 1, 1],
            [2, 1, 0],
        ]
        if attempt < len(init_patterns):
            pattern = init_patterns[attempt]
        else:
            pattern = [random.randint(0, 2) for _ in range(3)]
        
        for i, v in enumerate(pattern):
            colors[i] = v
        
        # Check if initial pattern is already valid for APs within first 3 positions
        valid_init = True
        for (a, b, cc, dd) in aps:
            if dd < 3:  # All indices in initial range
                vals = [colors[a], colors[b], colors[cc], colors[dd]]
                if len(set(vals)) < 3:
                    valid_init = False
                    break
        
        if not valid_init:
            continue
        
        result = _backtrack_search(colors, 3, aps, 3)
        if result is not None:
            # Verify
            if _validate(result):
                return result
    
    # If 3 colors fails, try 4 colors with more extensive search
    for attempt in range(100):
        if time.time() > DEADLINE:
            break
        
        random.seed(100000 + attempt * 777)
        colors = [None] * n
        for i in range(4):
            colors[i] = random.randint(0, 3)
        
        valid_init = True
        for (a, b, cc, dd) in aps:
            if dd < 4:
                vals = [colors[a], colors[b], colors[cc], colors[dd]]
                if len(set(vals)) < 3:
                    valid_init = False
                    break
        
        if not valid_init:
            continue
        
        result = _backtrack_search(colors, 4, aps, 4)
        if result is not None:
            if _validate(result):
                return result
    
    # Fallback: return a known valid coloring with more colors
    # Use a modular pattern that should work
    colors = []
    for x in range(1, n + 1):
        colors.append((x - 1) % 7)
    
    # Verify and if invalid, use greedy with more colors
    if not _validate(colors):
        # Greedy with 7 colors
        colors = [0]
        for x in range(2, n + 1):
            for c in range(7):
                ok = True
                for (a, b, cc, dd) in aps:
                    if x - 1 == dd or x - 1 == cc or x - 1 == b or x - 1 == a:
                        idxs = (a, b, cc, dd)
                        vals = []
                        for idx in idxs:
                            if idx == x - 1:
                                vals.append(c)
                            elif idx < len(colors):
                                vals.append(colors[idx])
                            else:
                                ok = False
                                break
                        if not ok:
                            break
                        if len(set(vals)) < 3:
                            ok = False
                            break
                if ok:
                    colors.append(c)
                    break
            else:
                colors.append(0)  # Should not happen
    
    return colors
