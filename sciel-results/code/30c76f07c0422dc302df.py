import time
import numpy as np

def _v2(n):
    """Return the 2-adic valuation of n (exponent of 2 in n)."""
    c = 0
    while n % 2 == 0:
        c += 1
        n //= 2
    return c

def _color_2adic(n):
    """Color position n (1-indexed) by v2(n) mod 3."""
    return _v2(n) % 3

def _count_violations(colors):
    """Count 4-APs that use fewer than 3 distinct colors."""
    n = len(colors)
    violations = 0
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i1 = a
            i2 = a + d
            i3 = a + 2*d
            i4 = a + 3*d
            if i4 >= n:
                break
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]
            c4 = colors[i4]
            if len({c1, c2, c3, c4}) < 3:
                violations += 1
    return violations

def _local_search_reduce_colors(colors, max_time=40.0):
    """
    Try to reduce the number of distinct colors by local search.
    If we have 3 colors and find a valid 2-coloring, return it.
    Otherwise, try to find a 3-coloring with better structure.
    Returns a valid coloring with minimal colors found.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    n = len(colors)
    best_colors = list(colors)
    best_num_colors = len(set(colors))
    
    # If we already have 3 colors, try to see if we can map to 2 colors
    # by checking if a 2-coloring exists via local search
    if best_num_colors <= 3:
        # Try simulated annealing to find a 2-coloring
        # Start from a random 2-coloring and try to fix violations
        import random
        rng = random.Random(42)
        
        # Try multiple random 2-colorings with local repair
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random 2-coloring
            c2 = [rng.randint(0, 1) for _ in range(n)]
            
            # Local search: flip positions to reduce violations
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random position to flip
                pos = rng.randint(0, n - 1)
                old_val = c2[pos]
                new_val = 1 - old_val
                
                # Count violations before and after
                # Only check APs involving pos
                old_viol = 0
                new_viol = 0
                
                for d in range(1, n):
                    # Check all 4-APs where pos could be one of the 4 terms
                    # pos = a, a+d, a+2d, a+3d
                    for role in range(4):
                        a = pos - role * d
                        if a < 0:
                            continue
                        i1 = a
                        i2 = a + d
                        i3 = a + 2*d
                        i4 = a + 3*d
                        if i4 >= n:
                            continue
                        
                        # Old colors
                        oc1 = c2[i1] if i1 != pos else old_val
                        oc2 = c2[i2] if i2 != pos else old_val
                        oc3 = c2[i3] if i3 != pos else old_val
                        oc4 = c2[i4] if i4 != pos else old_val
                        if len({oc1, oc2, oc3, oc4}) < 3:
                            old_viol += 1
                        
                        # New colors
                        nc1 = c2[i1] if i1 != pos else new_val
                        nc2 = c2[i2] if i2 != pos else new_val
                        nc3 = c2[i3] if i3 != pos else new_val
                        nc4 = c2[i4] if i4 != pos else new_val
                        if len({nc1, nc2, nc3, nc4}) < 3:
                            new_viol += 1
                
                # Accept if it doesn't increase violations
                if new_viol <= old_viol:
                    c2[pos] = new_val
                else:
                    # Simulated annealing: accept with some probability
                    # But for simplicity, just reject
                    pass
            
            # Check if c2 is valid
            if _count_violations(c2) == 0:
                return c2
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 128
    
    # Generate the 2-adic 3-coloring
    colors = [_color_2adic(i) for i in range(1, n + 1)]
    
    # Verify it's valid
    if _count_violations(colors) != 0:
        # Fallback: use more colors
        # Try greedy with 4 colors
        import random
        for seed in range(100):
            if time.time() > deadline - 10:
                break
            random.seed(seed)
            greedy_colors = []
            success = True
            for x in range(1, n + 1):
                valid_color = None
                for c in [0, 1, 2, 3]:
                    is_valid = True
                    for d in range(1, (x - 1) // 3 + 1):
                        if x - 3*d < 1:
                            break
                        c1 = greedy_colors[x - 1 - 3*d]
                        c2 = greedy_colors[x - 1 - 2*d]
                        c3 = greedy_colors[x - 1 - d]
                        if len({c, c1, c2, c3}) < 3:
                            is_valid = False
                            break
                    if is_valid:
                        valid_color = c
                        break
                if valid_color is None:
                    success = False
                    break
                greedy_colors.append(valid_color)
            if success and _count_violations(greedy_colors) == 0:
                colors = greedy_colors
                break
    
    # Try to reduce colors via local search
    if time.time() < deadline - 15:
        reduced = _local_search_reduce_colors(colors, max_time=40.0)
        if reduced is not None and _count_violations(reduced) == 0:
            colors = reduced
    
    # Remap to consecutive labels
    unique_colors = sorted(set(colors))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    remapped = [mapping[c] for c in colors]
    
    return remapped
