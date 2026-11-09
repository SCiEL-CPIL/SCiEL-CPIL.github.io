"""Optimized program: precomputed 4-AP constraints + stochastic local search on base-3 digits."""
import numpy as np

def _build_constraints():
    """Precompute all non-constant 4-term progressions in positions 1..128.
    Each progression is (a, a+d, a+2d, a+3d) with d>0, stored as 0-based indices.
    Returns a list of 4-tuples of indices.
    """
    progs = []
    for d in range(1, 43):  # d such that a+3d <= 128 => a <= 128-3d, a>=1 => need 128-3d>=1 => d<=42
        for a in range(1, 128 - 3 * d + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            progs.append((i0, i1, i2, i3))
    return progs

_CONSTRAINTS = _build_constraints()
_N = 128

def _check_validity(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors.
    colors: numpy array of ints, length 128.
    Returns True if valid.
    """
    for (i0, i1, i2, i3) in _CONSTRAINTS:
        c0, c1, c2, c3 = colors[i0], colors[i1], colors[i2], colors[i3]
        # Count distinct colors
        if c0 == c1 and c0 == c2 and c0 == c3:
            return False
        # Check if only 2 distinct colors
        # If not all same, check if exactly 2
        if c0 == c1:
            if c2 == c3 and c0 != c2:
                return False
            if c2 == c0:
                # c0=c1=c2, need c3 != c0 for 2 distinct? No, that's 2 distinct which is invalid
                # Actually c0=c1=c2, c3 different => 2 distinct colors => invalid
                return False
            if c3 == c0:
                # c0=c1=c3, c2 different => 2 distinct => invalid
                return False
            # c0=c1, c2 != c0, c3 != c0
            if c2 == c3:
                return False  # c0=c1, c2=c3, c0!=c2 => 2 distinct
            # else c0=c1, c2!=c0, c3!=c0, c2!=c3 => 3 distinct => ok
        elif c0 == c2:
            # c0=c2, c1!=c0
            if c3 == c0:
                return False  # c0=c2=c3, c1!=c0 => 2 distinct
            if c3 == c1:
                return False  # c0=c2, c1=c3, c0!=c1 => 2 distinct
            # else 3 distinct
        elif c0 == c3:
            # c0=c3, c1!=c0, c2!=c0
            if c1 == c2:
                return False  # c0=c3, c1=c2, c0!=c1 => 2 distinct
            # else 3 distinct
        elif c1 == c2:
            # c1=c2, c0!=c1, c3!=c1 (since c0!=c1 and we're in elif chain, c3 could be anything)
            if c3 == c0:
                return False  # c1=c2, c0=c3, c0!=c1 => 2 distinct
            if c3 == c1:
                return False  # c1=c2=c3, c0!=c1 => 2 distinct
            # else 3 distinct
        elif c1 == c3:
            # c1=c3, c0!=c1, c2!=c1
            if c0 == c2:
                return False  # c1=c3, c0=c2, c0!=c1 => 2 distinct
            # else 3 distinct
        elif c2 == c3:
            # c2=c3, c0!=c2, c1!=c2
            if c0 == c1:
                return False  # c2=c3, c0=c1, c0!=c2 => 2 distinct
            # else 3 distinct
        # all distinct => ok
    return True

def _count_colors(colors):
    """Return the number of distinct color labels used."""
    return len(set(colors.tolist()))

def construct():
    """Find a coloring of positions 1..128 with minimum number of colors
    such that every non-constant 4-term progression uses at least 3 distinct colors.
    Returns a list of 128 integer color labels in 0..127.
    """
    # Start with a known valid 3-coloring using a simple pattern, then try to reduce.
    # First, let's get a valid 3-coloring via greedy or a known construction.
    # Use the seed's greedy approach to get a starting valid coloring.
    colors_list = []
    for x in range(1, 129):
        assigned = False
        for color in range(3):
            ok = True
            # Check all 4-APs ending at position x-1 (0-based)
            for (i0, i1, i2, i3) in _CONSTRAINTS:
                if i3 == x - 1:
                    if i2 < len(colors_list):
                        c0 = colors_list[i0] if i0 < len(colors_list) else None
                        c1 = colors_list[i1] if i1 < len(colors_list) else None
                        c2 = colors_list[i2]
                        c3 = color
                        if c0 is None or c1 is None:
                            continue
                        distinct = len({c0, c1, c2, c3})
                        if distinct < 3:
                            ok = False
                            break
            if ok:
                colors_list.append(color)
                assigned = True
                break
        if not assigned:
            # Should not happen with 3 colors for 128, but fallback
            colors_list.append(0)
    
    colors = np.array(colors_list, dtype=np.int32)
    current_colors = _count_colors(colors)
    
    if current_colors <= 2:
        return colors_list
    
    # Stochastic local search: try to reduce number of colors
    # Strategy: try to recolor positions to use fewer distinct colors
    best_colors = colors.copy()
    best_count = current_colors
    
    rng = np.random.default_rng(42)
    
    # Try to find a valid 2-coloring (unlikely but check)
    # Then try to minimize colors among valid colorings
    
    # Local search: randomly pick a position and try changing its color to one of {0,1,2}
    # Accept if still valid and doesn't increase color count
    for iteration in range(50000):
        pos = rng.integers(0, _N)
        old_color = best_colors[pos]
        # Try each color
        for new_color in range(3):
            if new_color == old_color:
                continue
            best_colors[pos] = new_color
            if _check_validity(best_colors):
                new_count = _count_colors(best_colors)
                if new_count < best_count:
                    best_count = new_count
                    break
                # Same count: accept with some probability for exploration
                elif new_count == best_count and rng.random() < 0.1:
                    break
                else:
                    best_colors[pos] = old_color
            else:
                best_colors[pos] = old_color
    
    # Try to further reduce: if we have 3 colors, try to eliminate one
    # Identify which color is used least, and try to recolor those positions
    color_counts = {}
    for c in best_colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    if len(color_counts) > 1:
        # Try to eliminate the least-used color
        target_color = min(color_counts, key=color_counts.get)
        positions_to_change = [i for i in range(_N) if best_colors[i] == target_color]
        
        for pos in positions_to_change:
            old_color = best_colors[pos]
            for new_color in range(3):
                if new_color == old_color:
                    continue
                best_colors[pos] = new_color
                if _check_validity(best_colors):
                    new_count = _count_colors(best_colors)
                    if new_count < best_count:
                        best_count = new_count
                        break
                    else:
                        best_colors[pos] = old_color
    
    return best_colors.tolist()
