"""
Colorings with three colors in every four-term progression.
Structural change: use a constraint-propagation + randomized backtracking approach
with a different representation (bitmasks for color sets) and a global search
that tries to minimize the number of distinct colors used.
"""
import time
import random
import sys

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

# For each position, precompute the other 3 positions in each progression
# _OTHERS[pos][prog_idx] = (other1, other2, other3) as 0-indexed
_OTHERS = [[None] * _NUM_PROGS for _ in range(128)]
for idx, prog in enumerate(_PROGRESSIONS):
    a, b, c, d = prog[0]-1, prog[1]-1, prog[2]-1, prog[3]-1
    _OTHERS[a][idx] = (b, c, d)
    _OTHERS[b][idx] = (a, c, d)
    _OTHERS[c][idx] = (a, b, d)
    _OTHERS[d][idx] = (a, b, c)

def _count_distinct(colors, pos, new_color):
    """Count distinct colors in the progression if we assign new_color to pos.
    Returns the number of distinct colors among the 4 positions."""
    for p_idx in _POS_TO_PROGS[pos]:
        o1, o2, o3 = _OTHERS[pos][p_idx]
        c1, c2, c3 = colors[o1], colors[o2], colors[o3]
        # Check if any is unassigned
        if c1 == -1 or c2 == -1 or c3 == -1:
            continue
        # Count distinct
        s = {new_color, c1, c2, c3}
        if len(s) < 3:
            return 0  # invalid
    return 1  # valid

def _is_valid_fast(colors, pos, new_color):
    """Fast validity check using precomputed structures."""
    for p_idx in _POS_TO_PROGS[pos]:
        o1, o2, o3 = _OTHERS[pos][p_idx]
        c1, c2, c3 = colors[o1], colors[o2], colors[o3]
        if c1 == -1 or c2 == -1 or c3 == -1:
            continue
        # Need at least 3 distinct colors
        if c1 == c2 == c3:
            # All three same, so new_color must be different
            if new_color == c1:
                return False
        else:
            # At least two are different
            # Check if adding new_color keeps at least 3 distinct
            # The set is {new_color, c1, c2, c3}
            # We need len >= 3
            # This fails only if all four are in {x, y} for some x != y
            # i.e., the set has size <= 2
            # Since c1,c2,c3 are not all equal, the set {c1,c2,c3} has size 2 or 3
            # If size 3, adding new_color gives size >= 3, always valid
            # If size 2, say {x,y}, then new_color must not be in {x,y}
            # Wait, if {c1,c2,c3} = {x,y}, then {new_color, x, y} has size:
            #   2 if new_color in {x,y}
            #   3 if new_color not in {x,y}
            # So we need new_color not in {c1,c2,c3}
            distinct_others = set([c1, c2, c3])
            if len(distinct_others) == 2:
                if new_color in distinct_others:
                    return False
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
            if _is_valid_fast(colors, i, c):
                available.append(c)
        
        if not available:
            return None
        
        colors[i] = rng.choice(available)
    
    return colors

def _local_search_minimize_colors(colors, max_iterations=5000, seed=None):
    """Local search to minimize the number of distinct colors.
    Tries to recolor positions to reduce the number of distinct colors used."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n = 128
    current_colors = list(colors)
    current_nc = len(set(current_colors))
    best_colors = list(current_colors)
    best_nc = current_nc
    
    for it in range(max_iterations):
        if _time_left() < 1.0:
            break
        
        # Pick a random position
        pos = rng.randint(0, n-1)
        old_color = current_colors[pos]
        
        # Try to recolor to a color that is already used (to potentially reduce nc)
        used_colors = set(current_colors)
        
        # Try each used color
        for new_color in used_colors:
            if new_color == old_color:
                continue
            if _is_valid_fast(current_colors, pos, new_color):
                current_colors[pos] = new_color
                new_nc = len(set(current_colors))
                if new_nc < best_nc:
                    best_nc = new_nc
                    best_colors = list(current_colors)
                break
        else:
            # Try a random unused color (might help escape local minima)
            if rng.random() < 0.1:
                all_colors = set(range(128))
                unused = all_colors - used_colors
                if unused:
                    new_color = rng.choice(list(unused))
                    if _is_valid_fast(current_colors, pos, new_color):
                        current_colors[pos] = new_color
    
    return best_colors, best_nc

def _try_eliminate_color(colors, target_color, max_attempts=100, seed=None):
    """Try to eliminate target_color by recoloring its positions.
    Returns True if successful."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n = 128
    positions = [i for i in range(n) if colors[i] == target_color]
    if not positions:
        return True
    
    for attempt in range(max_attempts):
        if _time_left() < 1.0:
            break
        
        original = list(colors)
        rng.shuffle(positions)
        
        success = True
        for pos in positions:
            found = False
            candidates = [c for c in range(128) if c != target_color]
            rng.shuffle(candidates)
            
            for c_new in candidates:
                if _is_valid_fast(colors, pos, c_new):
                    colors[pos] = c_new
                    found = True
                    break
            
            if not found:
                colors[:] = original
                success = False
                break
        
        if success:
            return True
    
    return False

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_colors = None
    best_nc = 128
    
    # Strategy:
    # 1. Generate diverse initial colorings using greedy with different k values
    # 2. Apply local search to minimize colors
    # 3. Try to eliminate colors one by one
    
    # Try various k values and seeds
    k_values = [4, 5, 6, 7, 8, 10, 12, 16]
    seeds = range(20)
    
    for k in k_values:
        if _time_left() < 2.0:
            break
        
        for seed in seeds:
            if _time_left() < 1.0:
                break
            
            colors = _greedy_construct(k, seed=seed)
            if colors is None:
                continue
            
            nc = len(set(colors))
            
            if nc < best_nc or best_colors is None:
                # Improve via local search
                improved_colors, improved_nc = _local_search_minimize_colors(
                    colors, max_iterations=3000, seed=seed
                )
                
                if improved_nc < best_nc:
                    best_nc = improved_nc
                    best_colors = improved_colors
                
                # Try to eliminate colors
                current_colors = list(improved_colors)
                current_nc = improved_nc
                
                improved = True
                while improved and _time_left() > 2.0:
                    improved = False
                    used = sorted(set(current_colors))
                    rng = random.Random()
                    rng.shuffle(used)
                    
                    for tc in used:
                        if _time_left() < 1.0:
                            break
                        if tc in set(current_colors):
                            if _try_eliminate_color(current_colors, tc, max_attempts=50, seed=rng.randint(0, 10000)):
                                new_nc = len(set(current_colors))
                                if new_nc < current_nc:
                                    current_nc = new_nc
                                    improved = True
                                    if current_nc < best_nc:
                                        best_nc = current_nc
                                        best_colors = list(current_colors)
                                    break
                
                if best_nc == 3:
                    return best_colors
    
    # Fallback
    if best_colors is None:
        colors = _greedy_construct(128, seed=0)
        if colors:
            return colors
        else:
            return list(range(128))
    
    return best_colors
