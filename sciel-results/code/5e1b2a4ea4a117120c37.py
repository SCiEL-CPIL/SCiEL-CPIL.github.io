"""Colorings with three colors in every four-term progression."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _validity(c):
    """Check that every nonconstant 4-term progression uses >= 3 distinct colors."""
    n = len(c)
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b, cc, dd = a + d, a + 2 * d, a + 3 * d
            if dd >= n:
                break
            if len({c[a], c[b], c[cc], c[dd]}) < 3:
                return False
    return True


def _count_colors(c):
    return len(set(c))


def _violations(c):
    """Count how many 4-term progressions violate the >=3 distinct colors constraint."""
    n = len(c)
    v = 0
    for a in range(n):
        max_d = (n - a - 1) // 3
        for d in range(1, max_d + 1):
            b, cc, dd = a + d, a + 2 * d, a + 3 * d
            if dd >= n:
                break
            if len({c[a], c[b], c[cc], c[dd]}) < 3:
                v += 1
    return v


def _local_search(c, time_budget, rng):
    """Local search with simulated annealing to minimize color count."""
    c = list(c)
    n = len(c)
    best_c = c[:]
    best_count = _count_colors(c)
    
    # Precompute all 4-term progressions for fast violation checking
    progressions = []
    for a in range(n):
        max_d = (n - a - 1) // 3
        for d in range(1, max_d + 1):
            b, cc, dd = a + d, a + 2 * d, a + 3 * d
            if dd < n:
                progressions.append((a, b, cc, dd))
    
    # Start from 3 colors, try to reduce to 2
    # Strategy: iteratively try to recolor positions to reduce distinct colors
    # while maintaining validity
    
    t_start = time.time()
    deadline = t_start + time_budget
    
    # First, try simple greedy reduction
    # Sort colors by frequency (least frequent first) and try to eliminate
    for attempt in range(50):
        cur = c[:]
        cur_count = _count_colors(cur)
        if cur_count <= 2:
            break
        # Find least frequent color
        from collections import Counter
        cnt = Counter(cur)
        # Sort colors by count, try to eliminate the rarest
        colors_sorted = sorted(cnt.items(), key=lambda x: x[1])
        target_color = colors_sorted[0][0]
        # Try to recolor all positions with target_color
        positions = [i for i in range(n) if cur[i] == target_color]
        # Try to recolor each position to another color
        changed = True
        while changed and (time.time() < deadline):
            changed = False
            for i in positions[:]:
                if cur[i] != target_color:
                    continue
                # Try each other color
                for new_color in range(128):
                    if new_color == target_color:
                        continue
                    cur[i] = new_color
                    if _validity(cur):
                        positions = [p for p in positions if cur[p] == target_color]
                        changed = True
                        break
                    cur[i] = target_color
        c = cur[:]
        c_count = _count_colors(c)
        if c_count < best_count:
            best_count = c_count
            best_c = c[:]
    
    # Now do simulated annealing for further improvement
    if best_count > 2 and time.time() < deadline:
        cur = best_c[:]
        cur_count = _count_colors(cur)
        t = 1.0
        t_min = 0.001
        cooling = 0.999
        
        while time.time() < deadline and t > t_min:
            # Pick a random position and a random new color
            i = rng.randint(0, n - 1)
            old_color = cur[i]
            new_color = rng.randint(0, 127)
            if new_color == old_color:
                continue
            cur[i] = new_color
            new_count = _count_colors(cur)
            
            # Check validity
            valid = _validity(cur)
            
            if valid:
                # Accept if better or same
                if new_count <= cur_count:
                    if new_count < best_count:
                        best_count = new_count
                        best_c = cur[:]
                else:
                    # Worse in color count, accept with probability
                    delta = new_count - cur_count
                    if rng.random() < np.exp(-delta / t):
                        pass  # accept
                    else:
                        cur[i] = old_color
            else:
                # Invalid, revert
                cur[i] = old_color
            
            t *= cooling
    
    return best_c


def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    rng = random.Random(42)
    n = 128
    
    # Start with a known valid 3-coloring
    # Pattern: color by x mod 3, but verify it works
    # Actually, let's use a pattern that's more likely to be reducible
    # Color i (0-indexed) by (i // 4) % 3 -- blocks of 4
    # Or simpler: i % 3
    
    # Let's try multiple initial patterns and pick the best
    best_result = None
    best_score = 128  # max possible
    
    # Pattern 1: i % 3
    c1 = [(i) % 3 for i in range(n)]
    if _validity(c1):
        best_result = c1
        best_score = _count_colors(c1)
    
    # Pattern 2: (i // 2) % 3
    c2 = [(i // 2) % 3 for i in range(n)]
    if _validity(c2) and _count_colors(c2) < best_score:
        best_result = c2
        best_score = _count_colors(c2)
    
    # Pattern 3: i % 3 but shifted
    c3 = [(i + 1) % 3 for i in range(n)]
    if _validity(c3) and _count_colors(c3) < best_score:
        best_result = c3
        best_score = _count_colors(c3)
    
    # If no valid pattern found, use greedy construction
    if best_result is None:
        colors = []
        for x in range(1, n + 1):
            found = False
            for color in range(128):
                ok = True
                # Check all 4-term progressions ending at x-1
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 1 - 3 * d
                    b = x - 1 - 2 * d
                    cc = x - 1 - d
                    if a < 0:
                        continue
                    if len({color, colors[a], colors[b], colors[cc]}) < 3:
                        ok = False
                        break
                # Check all 4-term progressions where x-1 is not the last
                # Actually we need to check all progressions containing x-1
                # But for greedy, we only need to check ones where x-1 is the last element
                # because previous elements are already fixed
                if ok:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Shouldn't happen with 128 colors
                colors.append(0)
        best_result = colors
        best_score = _count_colors(best_result)
    
    # Now try to improve with local search
    if time.time() < deadline - 5:
        improved = _local_search(best_result, min(75.0, deadline - time.time() - 2), rng)
        if _count_colors(improved) < best_score:
            best_result = improved
            best_score = _count_colors(best_result)
    
    # Final validation
    if not _validity(best_result):
        # Fallback: use the original pattern
        best_result = [(i) % 3 for i in range(n)]
    
    return best_result

# EVOLVE-BLOCK-END
