import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 128
    num_colors = 3  # minimum possible
    
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            if a + 3 * d <= n:
                aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    # Try to find a 3-coloring using backtracking with constraint propagation
    # Use a smarter approach: try periodic patterns first, then random restarts
    
    def is_valid(coloring, idx):
        """Check if position idx (0-indexed) can have the given coloring without violating constraints."""
        # Check all 4-term APs that include position idx
        pos = idx + 1  # 1-indexed
        for (a, b, c, d) in aps:
            if pos in (a + 1, b + 1, c + 1, d + 1):
                # Check if all four positions are assigned
                vals = []
                for p in (a, b, c, d):
                    if coloring[p] is None:
                        break
                    vals.append(coloring[p])
                else:
                    if len(set(vals)) < 3:
                        return False
        return True
    
    def check_complete(coloring):
        """Check if a complete coloring is valid."""
        for (a, b, c, d) in aps:
            if len(set([coloring[a], coloring[b], coloring[c], coloring[d]])) < 3:
                return False
        return True
    
    # Strategy 1: Try a few structured patterns
    # Pattern: repeating block of length L
    for block_len in range(3, 20):
        for _ in range(100):
            if time.time() > deadline:
                break
            block = [random.randint(0, num_colors - 1) for _ in range(block_len)]
            coloring = [block[i % block_len] for i in range(n)]
            if check_complete(coloring):
                return coloring
    
    # Strategy 2: Backtracking search with randomization
    # Order positions by degree (number of APs they're in)
    pos_ap_count = [0] * n
    for (a, b, c, d) in aps:
        pos_ap_count[a] += 1
        pos_ap_count[b] += 1
        pos_ap_count[c] += 1
        pos_ap_count[d] += 1
    
    order = sorted(range(n), key=lambda i: -pos_ap_count[i])
    
    # Try multiple random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Shuffle color order for diversity
        color_order = list(range(num_colors))
        random.shuffle(color_order)
        
        coloring = [None] * n
        failed = False
        
        for idx in order:
            # Try each color
            valid_colors = []
            for c in color_order:
                coloring[idx] = c
                if is_valid(coloring, idx):
                    valid_colors.append(c)
                coloring[idx] = None
            
            if not valid_colors:
                failed = True
                break
            
            # Pick randomly among valid colors (with some heuristic)
            coloring[idx] = random.choice(valid_colors)
        
        if not failed and all(c is not None for c in coloring):
            if check_complete(coloring):
                return coloring
    
    # Fallback: greedy construction (like original but with 3 colors)
    colors = []
    for x in range(1, n + 1):
        placed = False
        for color in range(num_colors):
            valid = True
            # Check all 4-term APs containing position x
            for (a, b, c, d) in aps:
                pos = x
                if pos in (a + 1, b + 1, c + 1, d + 1):
                    vals = []
                    for p in (a, b, c, d):
                        if p + 1 == x:
                            vals.append(color)
                        elif p < x:
                            vals.append(colors[p - 1])
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                    if len(set(vals)) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                placed = True
                break
        if not placed:
            # Shouldn't happen with 3 colors if we're careful, but fallback
            colors.append(0)
    
    if check_complete(colors):
        return colors
    
    # Last resort: return a valid 4-coloring if 3-coloring fails
    # Use a known valid pattern with 4 colors
    colors4 = []
    for x in range(1, n + 1):
        colors4.append((x - 1) % 4)
    if check_complete(colors4):
        return colors4
    
    # Absolute fallback
    return list(range(n))
