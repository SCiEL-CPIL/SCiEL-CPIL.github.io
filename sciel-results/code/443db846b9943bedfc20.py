"""Rewritten program: greedy construction with local search to minimize color count."""
import time
import random
import numpy as np

def _validate(colors):
    """Check that every nonconstant 4-term progression uses at least 3 distinct colors."""
    n = len(colors)
    for i in range(n):
        for j in range(i+1, n):
            d = j - i
            # 4-term progression: i, j, j+d, j+2d
            k = j + d
            l = j + 2*d
            if l >= n:
                break
            # Nonconstant means d > 0, which is guaranteed since j > i
            c = {colors[i], colors[j], colors[k], colors[l]}
            if len(c) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_construct(max_colors, seed=None):
    """
    Greedy: build colors[0..127] one by one.
    At each position x (1-indexed), try colors 0..max_colors-1 in order.
    A color c is valid for position x if:
      - For every 4-term progression ending at x (i.e., x is the 4th term),
        the progression uses at least 3 distinct colors.
      - For every 3-term progression ending at x-1 or x-2 (x is 3rd or 4th term),
        we don't create a situation where a future 4-term progression could be forced
        to have only 2 colors... actually the seed checks two conditions:
        1. All 4-term progressions where x is the last term: need >=3 colors among the 4.
        2. All 3-term progressions where x is the 3rd term: the 3 terms must not all be the same
           (i.e., at least one of the previous two terms differs from x's color).
           Wait, let me re-read the seed.
    """
    if seed is not None:
        random.seed(seed)
    
    n = 128
    colors = [-1] * n
    
    for x in range(n):
        # x is 0-indexed, so position is x+1
        # We need to check all 4-term progressions where position x+1 is the 4th term
        # A 4-term progression: a, a+d, a+2d, a+3d where a+3d = x+1 => a = x+1-3d
        # So d ranges from 1 to x//3 (since a >= 1 => x+1-3d >= 1 => d <= x//3)
        
        # Also check 3-term progressions where position x+1 is the 3rd term:
        # a, a+d, a+2d where a+2d = x+1 => a = x+1-2d, d from 1 to x//2
        # The seed's second condition: any(colors[x-1-j*d] != color for j in (1,2))
        # This means: not both previous terms equal to color, i.e., at least one differs.
        # This ensures the 3-term progression isn't constant, which would force issues.
        
        found = False
        for color in range(max_colors):
            valid = True
            
            # Check 4-term progressions ending at position x+1
            for d in range(1, x // 3 + 1):
                a = x + 1 - 3 * d  # 1-indexed start
                if a < 1:
                    break
                # Positions: a, a+d, a+2d, a+3d = x+1
                # 0-indexed: a-1, a-1+d, a-1+2d, x
                c0 = colors[a-1]
                c1 = colors[a-1+d]
                c2 = colors[a-1+2*d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 3-term progressions ending at position x+1 (x is 3rd term)
            for d in range(1, x // 2 + 1):
                a = x + 1 - 2 * d  # 1-indexed start
                if a < 1:
                    break
                # Positions: a, a+d, a+2d = x+1
                # 0-indexed: a-1, a-1+d, x
                c0 = colors[a-1]
                c1 = colors[a-1+d]
                c2 = color
                # The seed condition: at least one of c0, c1 differs from c2
                if c0 == c2 and c1 == c2:
                    valid = False
                    break
            
            if valid:
                colors[x] = color
                found = True
                break
        
        if not found:
            return None
    
    return colors

def _local_search(colors, max_colors, time_budget, seed=None):
    """
    Try to reduce the number of colors used by reassigning colors.
    """
    if seed is not None:
        random.seed(seed)
    
    n = len(colors)
    current = list(colors)
    current_colors = set(current)
    current_count = len(current_colors)
    best = current[:]
    best_count = current_count
    
    start_time = time.time()
    
    # Try to eliminate one color at a time
    while time.time() - start_time < time_budget:
        # Find the least used color
        from collections import Counter
        cnt = Counter(current)
        # Sort colors by frequency
        color_freqs = sorted(cnt.items(), key=lambda x: x[1])
        
        eliminated_any = False
        for target_color, freq in color_freqs:
            if freq == 0:
                continue
            
            # Try to reassign all positions with target_color to other colors
            positions = [i for i in range(n) if current[i] == target_color]
            
            # Try each position in random order
            random.shuffle(positions)
            
            new_current = current[:]
            success = True
            
            for pos in positions:
                # Try to assign a different color
                assigned = False
                for new_color in range(max_colors):
                    if new_color == target_color:
                        continue
                    new_current[pos] = new_color
                    if _validate(new_current):
                        assigned = True
                        break
                    new_current[pos] = target_color  # restore
                if not assigned:
                    success = False
                    break
            
            if success:
                current = new_current
                current_colors = set(current)
                current_count = len(current_colors)
                if current_count < best_count:
                    best = current[:]
                    best_count = current_count
                eliminated_any = True
                break
        
        if not eliminated_any:
            # Try single-position swaps
            improved = False
            for _ in range(100):
                pos = random.randint(0, n-1)
                old_color = current[pos]
                for new_color in range(max_colors):
                    if new_color == old_color:
                        continue
                    current[pos] = new_color
                    if _validate(current):
                        new_count = len(set(current))
                        if new_count < best_count:
                            best = current[:]
                            best_count = new_count
                            improved = True
                        else:
                            current[pos] = old_color
                    else:
                        current[pos] = old_color
                if improved:
                    break
            
            if not improved:
                break
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # First, get a valid coloring with the greedy method using up to 128 colors
    # The seed uses up to 128 colors. Let's try to use fewer.
    
    # Try with increasing max_colors to find the minimum
    # But we want to minimize the number of colors actually used.
    
    # Start with a greedy using a reasonable number of colors
    best_result = None
    best_count = 128
    
    # Try greedy with different max_colors
    for max_c in [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 32, 64, 128]:
        if time.time() > deadline - 5:
            break
        
        result = _greedy_construct(max_c)
        if result is not None:
            cnt = _count_colors(result)
            if cnt < best_count:
                best_count = cnt
                best_result = result
    
    if best_result is None:
        # Fallback to the original seed approach
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3+1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                           for d in range(1, (x-1)//2+1))):
                    colors.append(color)
                    break
        return colors
    
    # Local search to try to reduce color count further
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        improved, improved_count = _local_search(best_result, best_count, remaining_time - 2)
        if improved_count < best_count:
            best_result = improved
            best_count = improved_count
    
    return best_result

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Colors used: {len(set(result))}, Valid: {_validate(result)}")
