"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _all_4aps(n):
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _check_validity_fast(colors, aps):
    for (a, b, c, d) in aps:
        ca, cb, cc, cd = colors[a-1], colors[b-1], colors[c-1], colors[d-1]
        # Count distinct
        if ca == cb == cc == cd:
            return False
        # Check if only 2 distinct
        distinct = set([ca, cb, cc, cd])
        if len(distinct) < 3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    n = 128
    start_time = time.time()
    aps = _all_4aps(n)
    
    best_result = None
    best_count = n + 1
    
    # Strategy: try structured colorings with few colors
    # Try patterns based on modular arithmetic
    
    # Try 3 colors with various patterns
    for max_colors in range(3, 12):
        if time.time() - start_time > 55:
            break
        
        # Try several random initializations and local search
        for seed in range(30):
            if time.time() - start_time > 55:
                break
            
            rng = random.Random(seed)
            colors = [rng.randint(0, max_colors - 1) for _ in range(n)]
            
            # Repair phase: fix violations
            for repair_iter in range(200):
                if time.time() - start_time > 55:
                    break
                
                violated = False
                for (a, b, c, d) in aps:
                    cs = {colors[a-1], colors[b-1], colors[c-1], colors[d-1]}
                    if len(cs) < 3:
                        violated = True
                        # Change one of the positions
                        idx = rng.choice([a-1, b-1, c-1, d-1])
                        old = colors[idx]
                        new = rng.randint(0, max_colors - 1)
                        while new == old:
                            new = rng.randint(0, max_colors - 1)
                        colors[idx] = new
                        break
                
                if not violated:
                    break
            
            # Check if valid
            if _check_validity_fast(colors, aps):
                cnt = _count_colors(colors)
                if cnt < best_count:
                    best_count = cnt
                    best_result = colors[:]
                
                # Try to reduce colors via remapping
                for reduce_iter in range(50):
                    if time.time() - start_time > 55:
                        break
                    
                    color_set = set(colors)
                    if len(color_set) <= 1:
                        break
                    
                    freq = Counter(colors)
                    sorted_colors = sorted(color_set, key=lambda c: freq[c])
                    
                    improved = False
                    for target in sorted_colors:
                        if freq[target] == 0:
                            continue
                        others = [c for c in color_set if c != target]
                        for new_c in others:
                            new_colors = colors[:]
                            for i in range(n):
                                if new_colors[i] == target:
                                    new_colors[i] = new_c
                            if _check_validity_fast(new_colors, aps):
                                colors = new_colors
                                cnt = _count_colors(colors)
                                if cnt < best_count:
                                    best_count = cnt
                                    best_result = colors[:]
                                improved = True
                                break
                        if improved:
                            break
                    
                    if not improved:
                        break
        
        if best_count <= max_colors:
            break
    
    if best_result is None:
        # Fallback
        colors = []
        for x in range(1, n + 1):
            for color in range(n):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    prev = [colors[x-1-j*d] for j in (1, 2, 3)]
                    if len(set([color] + prev)) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    break
            else:
                colors.append(0)
        best_result = colors
    
    return [min(c, 127) for c in best_result]
# EVOLVE-BLOCK-END
