"""Combined program: constraint-based construction + randomized local search for color minimization."""
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

def _greedy_construct(max_colors=4, seed=0):
    """Greedy construction trying to use at most max_colors colors."""
    rng = random.Random(seed)
    colors = [0] * 128
    for x in range(128):
        # Try each color from 0 to max_colors-1 in random order
        color_order = list(range(max_colors))
        rng.shuffle(color_order)
        found = False
        for color in color_order:
            colors[x] = color
            # Check all 4-term progressions ending at or including position x
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

def _try_remove_color(colors, target_color, n=128):
    """Try to eliminate target_color by recoloring its positions to other colors."""
    other_colors = sorted(set(colors) - {target_color})
    if not other_colors:
        return None
    
    positions = [i for i in range(n) if colors[i] == target_color]
    if not positions:
        return list(colors)
    
    new_colors = list(colors)
    # Sort positions by degree of constraint (most constrained first)
    pos_constraint = []
    for pos in positions:
        # Count 4-APs involving this position
        count = 0
        for a in range(max(0, pos - 3*42), min(n, pos + 1)):
            for d in range(1, 43):
                b = a + d
                c = a + 2*d
                e = a + 3*d
                if e >= n:
                    break
                if pos in (a, b, c, e):
                    count += 1
        pos_constraint.append((count, pos))
    pos_constraint.sort(reverse=True)
    
    for _, pos in pos_constraint:
        found = False
        for new_color in other_colors:
            new_colors[pos] = new_color
            # Check validity for 4-APs involving this position
            valid = True
            for a in range(max(0, pos - 3*42), min(n, pos + 1)):
                for d in range(1, 43):
                    b = a + d
                    c = a + 2*d
                    e = a + 3*d
                    if e >= n:
                        break
                    cols = {new_colors[a], new_colors[b], new_colors[c], new_colors[e]}
                    if len(cols) < 3:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    
    return new_colors

def _local_search_reduce_colors(colors, deadline, max_attempts=100):
    """Local search to minimize number of colors used by removing colors one at a time."""
    colors = list(colors)
    best_count = _count_colors(colors)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        unique_colors = sorted(set(colors))
        if len(unique_colors) <= 2:
            break
        
        improved = False
        # Try removing each color in random order
        rng = random.Random(attempt)
        color_order = list(unique_colors)
        rng.shuffle(color_order)
        
        for target in color_order:
            if time.time() > deadline:
                break
            new_colors = _try_remove_color(colors, target)
            if new_colors is not None and _check_valid(new_colors):
                colors = new_colors
                best_count = _count_colors(colors)
                improved = True
                break
        
        if not improved:
            break
    
    return colors

def _randomized_greedy(K, seed, n=128):
    """Randomized greedy with preference for already-used colors."""
    rng = random.Random(seed)
    colors = [None] * n
    used_colors = set()
    
    for i in range(n):
        pos = i + 1
        valid = set(range(K))
        
        # Check all 4-APs ending at pos
        max_d = (pos - 1) // 3
        for d in range(1, max_d + 1):
            i1 = pos - 3*d
            i2 = pos - 2*d
            i3 = pos - d
            
            c1 = colors[i1-1]
            c2 = colors[i2-1]
            c3 = colors[i3-1]
            
            for c in list(valid):
                if len({c1, c2, c3, c}) < 3:
                    valid.discard(c)
        
        if not valid:
            return None
        
        # Bias towards used colors to minimize palette
        preferred = valid & used_colors
        if preferred:
            c = rng.choice(list(preferred))
        else:
            c = rng.choice(list(valid))
        
        colors[i] = c
        used_colors.add(c)
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10s safety margin
    
    n = 128
    best_solution = None
    best_count = 128
    
    # Phase 1: Try randomized greedy with K=3,4,5...
    for K in range(3, 10):
        if time.time() > deadline:
            break
        
        max_restarts = 2000 if K == 3 else 500
        found = False
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            colors = _randomized_greedy(K, seed=r, n=n)
            if colors is not None and _check_valid(colors):
                count = _count_colors(colors)
                if count < best_count:
                    best_count = count
                    best_solution = list(colors)
                    found = True
                # If we found a 3-coloring, that's likely optimal
                if count == 3:
                    break
        
        if found and best_count == 3:
            break
    
    # Phase 2: If no solution yet, try basic greedy
    if best_solution is None:
        for max_colors in range(3, 10):
            if time.time() > deadline:
                break
            colors = _greedy_construct(max_colors, seed=0)
            if colors is not None and _check_valid(colors):
                best_solution = colors
                best_count = _count_colors(colors)
                break
    
    # Phase 3: Local search to reduce colors further
    if best_solution is not None and time.time() < deadline - 5:
        improved = _local_search_reduce_colors(best_solution, deadline)
        if improved is not None and _check_valid(improved):
            if _count_colors(improved) <= best_count:
                best_solution = improved
                best_count = _count_colors(best_solution)
    
    # Fallback: use original greedy approach if time remains
    if best_solution is None and time.time() < deadline - 10:
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color)
                    break
        return colors
    
    # Emergency fallback
    if best_solution is None:
        return list(range(128))
    
    return best_solution
# EVOLVE-BLOCK-END
