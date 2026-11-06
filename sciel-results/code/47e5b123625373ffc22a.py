"""Improved search: fast greedy with bitmask/set checks and color elimination local search."""
import time
import random

_DEADLINE = 90.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed() - 5.0

def _generate_progressions(n=128):
    """Generate all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) with 1-indexed positions."""
    progs = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            progs.append((a, a + d, a + 2 * d, a + 3 * d))
    return progs

# Precompute progressions and incidence
_PROGRESSIONS = _generate_progressions(128)
_NUM_PROGS = len(_PROGRESSIONS)

# Map position (0-indexed) to list of progressions (indices into _PROGRESSIONS)
_POS_TO_PROGS = [[] for _ in range(128)]
for idx, prog in enumerate(_PROGRESSIONS):
    for p in prog:
        _POS_TO_PROGS[p-1].append(idx)

def _is_valid_assignment(pos, color, colors):
    """Check if assigning 'color' to 'pos' (0-indexed) is valid given current 'colors'.
    'colors' is a list of length 128. Unassigned positions are -1.
    """
    for p_idx in _POS_TO_PROGS[pos]:
        a, b, c, d = _PROGRESSIONS[p_idx]
        # Get the other 3 positions
        if pos == a - 1:
            others = [b - 1, c - 1, d - 1]
        elif pos == b - 1:
            others = [a - 1, c - 1, d - 1]
        elif pos == c - 1:
            others = [a - 1, b - 1, d - 1]
        else: # pos == d - 1
            others = [a - 1, b - 1, c - 1]
        
        # Check if all others are assigned
        all_assigned = True
        for o in others:
            if colors[o] == -1:
                all_assigned = False
                break
        
        if not all_assigned:
            continue
            
        # Check distinct colors count
        # We have colors[others] and the new 'color'
        # We need at least 3 distinct colors in total
        # Set of colors: {color} U {colors[o] for o in others}
        # Optimization: 
        # If the 3 others are all same color x:
        #   Set is {color, x}. Size <= 2. Always invalid.
        # If the 3 others are {x, x, y}:
        #   Set is {color, x, y}. 
        #   If color == x or color == y, size is 2. Invalid.
        #   If color != x and color != y, size is 3. Valid.
        # If the 3 others are {x, y, z} (all distinct):
        #   Set is {color, x, y, z}. Size >= 3. Always valid.
        
        c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
        
        if c1 == c2 == c3:
            return False # No color can fix this
        
        if c1 == c2 or c1 == c3 or c2 == c3:
            # Two are same, one is different
            # Find the unique color
            if c1 == c2:
                unique = c3
                duplicate = c1
            elif c1 == c3:
                unique = c2
                duplicate = c1
            else: # c2 == c3
                unique = c1
                duplicate = c2
            
            # If new color is duplicate or unique, total distinct is 2.
            # If new color is different from both, total distinct is 3.
            if color == duplicate or color == unique:
                return False
        
        # If all 3 are distinct, any color is valid (adds 3rd or 4th distinct)
        
    return True

def _greedy_construct(k, seed=None):
    """Greedy construction using at most k colors. Returns coloring or None."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n = 128
    colors = [-1] * n
    
    for i in range(n):
        available = []
        for c in range(k):
            if _is_valid_assignment(i, c, colors):
                available.append(c)
        
        if not available:
            return None
        
        colors[i] = rng.choice(available)
    
    return colors

def _eliminate_color(colors, target_color, k, rng):
    """Try to eliminate 'target_color' from 'colors'.
    Returns True if successful (color no longer present), False otherwise.
    Modifies 'colors' in place if successful.
    """
    n = 128
    positions = [i for i in range(n) if colors[i] == target_color]
    if not positions:
        return True # Already eliminated
    
    rng.shuffle(positions)
    
    # We need to be able to backtrack if we get stuck
    # Simple approach: copy colors, try to modify, if fail, restore
    original_colors = list(colors)
    
    for pos in positions:
        found = False
        # Try other colors
        # We can try colors in random order or sorted
        candidates = [c for c in range(k) if c != target_color]
        rng.shuffle(candidates)
        
        for c_new in candidates:
            if _is_valid_assignment(pos, c_new, colors):
                colors[pos] = c_new
                found = True
                break
        
        if not found:
            # Failed to reassign this position
            colors[:] = original_colors
            return False
    
    return True

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_colors = None
    best_nc = 128
    
    # Try increasing number of colors for greedy
    # Start with small k to find a valid solution quickly
    # Then use elimination to reduce
    
    for k in range(3, 10):
        if _time_left() < 1.0:
            break
            
        # Try multiple seeds
        for seed in range(50):
            if _time_left() < 0.5:
                break
                
            colors = _greedy_construct(k, seed=seed)
            if colors is None:
                continue
                
            nc = len(set(colors))
            
            # Try to reduce colors
            if nc < best_nc or best_colors is None:
                # We have a valid coloring. Try to improve.
                current_colors = list(colors)
                current_nc = nc
                
                # Loop to eliminate colors
                # We can try to eliminate any color present
                # To be efficient, try to eliminate the least frequent first?
                # Or just iterate.
                
                improved = True
                while improved and _time_left() > 1.0:
                    improved = False
                    used_colors = set(current_colors)
                    # Try to eliminate each used color
                    # Shuffle order to avoid bias
                    colors_to_try = list(used_colors)
                    random.shuffle(colors_to_try)
                    
                    for tc in colors_to_try:
                        if _time_left() < 0.5:
                            break
                        # Only try if it's still present
                        if tc in set(current_colors):
                            # Attempt elimination
                            if _eliminate_color(current_colors, tc, k, random):
                                new_nc = len(set(current_colors))
                                if new_nc < current_nc:
                                    current_nc = new_nc
                                    improved = True
                                    # Update best if better
                                    if current_nc < best_nc:
                                        best_nc = current_nc
                                        best_colors = list(current_colors)
                                    # Break to restart the while loop with new set of colors
                                    break
                    
                    if current_nc < best_nc:
                        best_nc = current_nc
                        best_colors = list(current_colors)

            # If we found a very good solution (e.g., 3 colors), we might stop early
            if best_nc == 3:
                # 3 is the theoretical minimum? 
                # "At least three distinct colors". 
                # If we use exactly 3 colors, it's optimal.
                # Can we use 2? No, because max distinct is 2.
                # So 3 is the minimum possible.
                return best_colors

    if best_colors is None:
        # Fallback: return a valid coloring with many colors if we couldn't find a good one
        # This should not happen with k up to 9
        # Just return a simple valid coloring
        # Greedy with k=128 should always work?
        colors = _greedy_construct(128, seed=0)
        if colors:
            return colors
        else:
            return list(range(128))

    return best_colors
