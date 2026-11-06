import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit

    n = 128

    # Precompute all 4-term progressions (a, a+d, a+2d, a+3d) with 0-based indices
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l < n:
                progressions.append((i, j, k, l))

    m = len(progressions)

    # For each position, store indices of progressions it belongs to
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)

    def count_violations_fast(colors):
        """Fast violation count: count progressions with < 3 distinct colors."""
        v = 0
        for (a, b, c, d) in progressions:
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Quick check using set
            if len({ca, cb, cc, cd}) < 3:
                v += 1
        return v

    def is_valid(colors):
        return count_violations_fast(colors) == 0

    def relabel(colors):
        """Relabel so colors are 0..k-1."""
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def greedy_forward():
        """Greedy forward construction: pick smallest valid color at each position."""
        colors = [0] * n
        
        for i in range(n):
            forbidden = set()
            
            # i as 4th term: a, a+d, a+2d, i where i = a+3d => a = i-3d
            for d in range(1, i // 3 + 1):
                a = i - 3 * d
                if a >= 0:
                    prev3 = {colors[a], colors[a + d], colors[a + 2 * d]}
                    if len(prev3) <= 2:
                        forbidden |= prev3
            
            # i as 3rd term: a, a+d, i, a+3d where i = a+2d => a = i-2d
            for d in range(1, min(i // 2, (n - 1 - i)) + 1):
                a = i - 2 * d
                if a >= 0:
                    b = i - d
                    e = i + d
                    if e < n:
                        s = {colors[a], colors[b], colors[e]}
                        if len(s) <= 2:
                            forbidden |= s
            
            # i as 2nd term: a, i, a+2d, a+3d where i = a+d => a = i-d
            for d in range(1, min(i, (n - 1 - i) // 2) + 1):
                a = i - d
                if a >= 0:
                    c = i + d
                    e = i + 2 * d
                    if e < n:
                        s = {colors[a], colors[c], colors[e]}
                        if len(s) <= 2:
                            forbidden |= s
            
            # i as 1st term: i, i+d, i+2d, i+3d
            for d in range(1, (n - 1 - i) // 3 + 1):
                b = i + d
                c = i + 2 * d
                e = i + 3 * d
                s = {colors[b], colors[c], colors[e]}
                if len(s) <= 2:
                    forbidden |= s

            # Pick smallest non-forbidden color
            for c in range(n):
                if c not in forbidden:
                    colors[i] = c
                    break
            else:
                colors[i] = 0
        
        return colors

    def local_search_reduce(colors, max_iters=50000, seed=42):
        """Local search to reduce the number of distinct colors while maintaining validity."""
        rng = random.Random(seed)
        colors = colors[:]
        n_colors = len(set(colors))
        
        if n_colors <= 3:
            return colors, n_colors

        # Track which colors are in use
        colors_in_use = sorted(set(colors))
        
        # Try to merge color classes by recoloring
        best_colors = colors[:]
        best_n_colors = n_colors
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random color to try to eliminate
            target = rng.choice(colors_in_use)
            positions = [i for i in range(n) if colors[i] == target]
            if not positions:
                continue
            
            # Try to recolor all positions with 'target' to other colors
            other_colors = [c for c in colors_in_use if c != target]
            if not other_colors:
                break
            
            success = True
            temp_colors = colors[:]
            
            for pos in positions:
                found = False
                for new_c in other_colors:
                    temp_colors[pos] = new_c
                    # Check if this assignment creates any violations
                    valid = True
                    for idx in prog_at[pos]:
                        a, b, c, d = progressions[idx]
                        ca, cb, cc, cd = temp_colors[a], temp_colors[b], temp_colors[c], temp_colors[d]
                        if len({ca, cb, cc, cd}) < 3:
                            valid = False
                            break
                    if valid:
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                colors = temp_colors
                colors_in_use = sorted(set(colors))
                n_colors = len(colors_in_use)
                if n_colors < best_n_colors:
                    best_n_colors = n_colors
                    best_colors = colors[:]
                if n_colors <= 3:
                    break
        
        return best_colors, best_n_colors

    # Step 1: Greedy forward construction
    colors = greedy_forward()
    
    if not is_valid(colors):
        # Fallback: use many colors
        colors = list(range(n))
        return colors

    colors, n_colors = relabel(colors)
    best_colors = colors[:]
    best_n_colors = n_colors
    
    # Step 2: Try to reduce colors via local search with multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        result, nc = local_search_reduce(best_colors, max_iters=20000, seed=seed)
        if nc < best_n_colors:
            best_n_colors = nc
            best_colors = result
            if best_n_colors <= 3:
                break

    # Relabel final result
    final_colors, _ = relabel(best_colors)
    
    # Final validation
    if not is_valid(final_colors):
        # Should not happen, but fallback
        final_colors = list(range(n))

    return final_colors
