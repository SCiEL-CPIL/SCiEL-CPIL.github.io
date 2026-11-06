import time
import random
from itertools import product

def _precompute_4aps(n):
    """Precompute all non-constant 4-APs as tuples of indices."""
    aps = []
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            # Non-constant: d > 0 ensures this, but let's verify
            # A 4-AP (a, a+d, a+2d, a+3d) is constant only if d=0, 
            # but d starts from 1, so all are non-constant
            aps.append((a, b, c, e))
    return aps

def _validate_with_aps(colors, aps):
    """Check if every 4-AP uses at least 3 distinct colors."""
    for a, b, c, e in aps:
        cols = {colors[a], colors[b], colors[c], colors[e]}
        if len(cols) < 3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_with_constraints(n=128, seed=0, num_colors=3):
    """Greedy coloring with constraint checking using precomputed APs."""
    rng = random.Random(seed)
    colors = [None] * n
    aps = _precompute_4aps(n)
    
    # For each position, precompute which APs it's part of
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)
    
    for i in range(n):
        tried = False
        for color in rng.sample(range(num_colors), num_colors):
            colors[i] = color
            # Check all APs that include position i and have all positions assigned
            valid = True
            for ap_idx in pos_aps[i]:
                a, b, c, e = aps[ap_idx]
                if colors[a] is None or colors[b] is None or colors[c] is None or colors[e] is None:
                    continue
                cols = {colors[a], colors[b], colors[c], colors[e]}
                if len(cols) < 3:
                    valid = False
                    break
            if valid:
                tried = True
                break
        if not tried:
            return None
    return colors

def _backtrack_search(n=128, num_colors=3, max_time=30, seed=0):
    """Backtracking search for a valid coloring."""
    start_time = time.time()
    rng = random.Random(seed)
    colors = [None] * n
    aps = _precompute_4aps(n)
    
    # For each position, precompute which APs it's part of
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)
    
    # Order positions by degree (number of APs they're in) for better pruning
    order = list(range(n))
    order.sort(key=lambda i: len(pos_aps[i]), reverse=True)
    
    def backtrack(pos_idx):
        if time.time() - start_time > max_time:
            return False
        if pos_idx == n:
            return True
        
        i = order[pos_idx]
        for color in rng.sample(range(num_colors), num_colors):
            colors[i] = color
            # Check all APs that include position i and have all other positions assigned
            valid = True
            for ap_idx in pos_aps[i]:
                a, b, c, e = aps[ap_idx]
                if colors[a] is None or colors[b] is None or colors[c] is None or colors[e] is None:
                    continue
                cols = {colors[a], colors[b], colors[c], colors[e]}
                if len(cols) < 3:
                    valid = False
                    break
            if valid:
                if backtrack(pos_idx + 1):
                    return True
            colors[i] = None
        return False
    
    if backtrack(0):
        return colors
    return None

def _local_search_improved(colors, n=128, max_iters=10000, seed=0, aps=None):
    """Improve a valid coloring by trying to reduce color count."""
    if aps is None:
        aps = _precompute_4aps(n)
    rng = random.Random(seed)
    current = list(colors)
    current_colors = set(current)
    
    # Precompute APs per position
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[e].append(idx)
    
    best = list(current)
    best_count = _count_colors(current)
    
    for _ in range(max_iters):
        # Try to remove a color by recoloring positions with that color
        if len(current_colors) <= 3:
            break
        target_color = rng.choice(list(current_colors))
        positions = [i for i in range(n) if current[i] == target_color]
        
        new_current = list(current)
        success = True
        for pos in positions:
            other_colors = [c for c in current_colors if c != target_color]
            found = False
            for new_color in rng.sample(other_colors, len(other_colors)):
                new_current[pos] = new_color
                valid = True
                for ap_idx in pos_aps[pos]:
                    a, b, c, e = aps[ap_idx]
                    cols = {new_current[a], new_current[b], new_current[c], new_current[e]}
                    if len(cols) < 3:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if not found:
                success = False
                break
        
        if success:
            current = new_current
            current_colors = set(current)
            if _count_colors(current) < best_count:
                best = list(current)
                best_count = _count_colors(current)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 128
    aps = _precompute_4aps(n)
    
    # Strategy 1: Try to find a 3-coloring via backtracking
    best = None
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        result = _backtrack_search(n, 3, max_time=5, seed=seed)
        if result is not None:
            if _validate_with_aps(result, aps):
                best = result
                break
    
    # Strategy 2: If no 3-coloring found, try greedy with 3 colors
    if best is None:
        for seed in range(100):
            if time.time() > deadline - 10:
                break
            result = _greedy_with_constraints(n, seed, num_colors=3)
            if result is not None:
                if _validate_with_aps(result, aps):
                    best = result
                    break
    
    # Strategy 3: If still no 3-coloring, try 4 colors
    if best is None:
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            result = _greedy_with_constraints(n, seed, num_colors=4)
            if result is not None:
                if _validate_with_aps(result, aps):
                    best = result
                    break
    
    # If we have a valid coloring, try to improve it
    if best is not None:
        if time.time() < deadline - 3:
            improved = _local_search_improved(best, n, max_iters=5000, seed=42, aps=aps)
            if _validate_with_aps(improved, aps) and _count_colors(improved) <= _count_colors(best):
                best = improved
    
    if best is not None:
        return best
    
    # Fallback: simple greedy with more colors
    colors = []
    for x in range(1, n + 1):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3*d
                b = x - 1 - 2*d
                c = x - 1 - d
                if a < 0:
                    break
                cols = {color, colors[a], colors[b], colors[c]}
                if len(cols) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
    
    return colors
