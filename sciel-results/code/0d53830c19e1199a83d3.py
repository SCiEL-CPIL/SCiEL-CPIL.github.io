import random
import time

def construct():
    n = 128
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a3 - 1))
    
    start_time = time.time()
    time_limit = 55  # seconds
    
    # Start with a valid coloring using n colors
    colors = list(range(n))
    best_colors = colors[:]
    best_count = n
    
    # Local search: try to merge colors
    for iteration in range(1000000):
        if time.time() - start_time > time_limit:
            break
        
        c1 = random.randint(0, n - 1)
        c2 = random.randint(0, n - 1)
        if c1 == c2:
            continue
        
        # Only try to merge if both colors exist
        if c1 not in colors or c2 not in colors:
            continue
        
        # Try merging c2 into c1
        new_colors = colors[:]
        for i in range(n):
            if new_colors[i] == c2:
                new_colors[i] = c1
        
        if _is_valid(new_colors, aps):
            colors = new_colors
            count = len(set(colors))
            if count < best_count:
                best_count = count
                best_colors = colors[:]
    
    return best_colors

def _is_valid(colors, aps):
    for (i, j, k, l) in aps:
        distinct = len({colors[i], colors[j], colors[k], colors[l]})
        if distinct < 3:
            return False
    return True
