"""Colorings with three colors in every four-term progression."""

import random
import time


def construct():
    """
    Construct a coloring of positions 1..128 with integer labels in 0..127 such that
    every nonconstant four-term arithmetic progression uses at least three distinct colors.
    Minimize the number of distinct labels actually used.
    """
    N = 128
    best_colors = None
    best_count = N + 1
    
    # Try different numbers of colors from 3 upward
    for num_colors in range(3, 10):
        if num_colors >= best_count:
            break
        
        # Try multiple random assignments
        found = False
        for attempt in range(200):
            random.seed(attempt)
            colors = [random.randint(0, num_colors - 1) for _ in range(N)]
            
            if _is_valid(color):
                pass  # placeholder, actual check below
            if _is_valid(color):
                found = True
                break
        
        # Use a proper validation
        for attempt in range(500):
            rng = random.Random(attempt)
            colors = [rng.randint(0, num_colors - 1) for _ in range(N)]
            if _is_valid(color):
                break
            if _is_valid(colors):
                if best_colors is None or num_colors < best_count:
                    best_colors = colors
                    best_count = num_colors
                found = True
                break
        
        if found:
            break
    
    if best_colors is None:
        # Fallback: greedy construction
        best_colors = _greedy_construct(N)
        best_count = len(set(best_colors))
    
    return best_colors


def _is_valid(colors):
    """Check if every nonconstant 4-term AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            vals = {colors[a], colors[b], colors[c], colors[e]}
            if len(vals) < 3:
                return False
    return True


def _greedy_construct(N):
    """Greedy construction: assign smallest possible color at each position."""
    colors = []
    for x in range(1, N + 1):
        idx = x - 1
        # Try colors 0, 1, 2, ...
        for color in range(N):
            valid = True
            # Check all 4-term APs ending at position x
            for d in range(1, (x - 1) // 3 + 1):
                # AP: x-3d, x-2d, x-d, x
                i0 = x - 1 - 3 * d
                i1 = x - 1 - 2 * d
                i2 = x - 1 - d
                if i0 < 0:
                    continue
                vals = {color, colors[i0], colors[i1], colors[i2]}
                if len(vals) < 3:
                    valid = False
                    break
            # Also check APs where x is not the last element but part of a 4-term AP
            # Actually we need to check all 4-term APs that include position x
            # For simplicity, only check those ending at x (greedy ensures previous positions are fine)
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)  # should not happen
    return colors


# Fix the typo in _is_valid call and clean up
def _is_valid(color):
    """Check if every nonconstant 4-term AP uses at least 3 distinct colors."""
    n = len(color)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            vals = {color[a], color[b], color[c], color[e]}
            if len(vals) < 3:
                return False
    return True
