import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = 128

    # 1. Run Seed Greedy Logic
    colors_seed = []
    for x in range(1, n + 1):
        found = False
        for color in range(n):  # Max 128 colors
            # Check constraints for position x (1-indexed)
            # Indices in colors_seed are 0..x-2
            # We are adding color to index x-1
            valid = True
            # 4-APs ending at x
            # a, a+d, a+2d, a+3d = x
            # a+3d <= x => d <= (x-1)//3
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                # Positions: a, a+d, a+2d, x
                # Indices: a-1, a+d-1, a+2d-1, x-1
                c1 = colors_seed[a - 1]
                c2 = colors_seed[a + d - 1]
                c3 = colors_seed[a + 2 * d - 1]
                c4 = color
                if len({c1, c2, c3, c4}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # 3-APs ending at x (No constant)
            # a, a+d, a+2d = x
            # a+2d <= x => d <= (x-1)//2
            for d in range(1, (x - 1) // 2 + 1):
                a = x - 2 * d
                # Positions: a, a+d, x
                # Indices: a-1, a+d-1, x-1
                c1 = colors_seed[a - 1]
                c2 = colors_seed[a + d - 1]
                c3 = color
                if c1 == c2 == c3:
                    valid = False
                    break
            
            if valid:
                colors_seed.append(color)
                found = True
                break
        if not found:
            # Should not happen if n is large enough palette
            colors_seed.append(0)
            
    count_seed = len(set(colors_seed))
    best_colors = colors_seed
    best_count = count_seed

    # 2. Run Search for smaller k
    # Precompute constraints for 0-indexed
    end_4aps = [[] for _ in range(n)]
    for x in range(n):
        for d in range(1, x // 3 + 1):
            a = x - 3 * d
            if a >= 0:
                end_4aps[x].append((a, d))
                
    end_3aps = [[] for _ in range(n)]
    for x in range(n):
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            if a >= 0:
                end_3aps[x].append((a, d))

    def is_valid_color(x, color, colors, k):
        for a, d in end_4aps[x]:
            c1 = colors[a]
            c2 = colors[a + d]
            c3 = colors[a + 2 * d]
            c4 = color
            if len({c1, c2, c3, c4}) < 3:
                return False
        for a, d in end_3aps[x]:
            c1 = colors[a]
            c2 = colors[a + d]
            c3 = color
            if c1 == c2 == c3:
                return False
        return True

    def dfs(x, colors, k, limit):
        if x == n:
            return True
        if time.time() > limit:
            return False
        
        colors_to_try = list(range(k))
        random.shuffle(colors_to_try)
        
        for c in colors_to_try:
            if is_valid_color(x, c, colors, k):
                colors[x] = c
                if dfs(x + 1, colors, k, limit):
                    return True
                colors[x] = -1
        return False

    # Try k from 3 to best_count - 1
    for k in range(3, best_count):
        if time.time() > deadline:
            break
        
        # Try a few restarts
        for _ in range(10):
            if time.time() > deadline:
                break
            colors = [-1] * n
            if dfs(0, colors, k, deadline):
                if k < best_count:
                    best_colors = colors
                    best_count = k
                break # Found valid k, move to next k (smaller)
        
        if best_count == k:
            break

    return best_colors
