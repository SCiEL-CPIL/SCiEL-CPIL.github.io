import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 128
    
    # Precompute APs ending at each position for fast checking
    # aps_ending[i] = list of (a, d) such that a+3d = i
    aps_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            if a >= 0:
                aps_ending[i].append((a, d))
                
    def is_valid(colors):
        """Check if the coloring is valid for all 4-term APs."""
        for i in range(n):
            for a, d in aps_ending[i]:
                vals = {colors[a], colors[a+d], colors[a+2*d], colors[i]}
                if len(vals) < 3:
                    return False
        return True

    def try_greedy(k, max_restarts=2000):
        """Try to find a valid coloring with k colors using randomized greedy."""
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            colors = [None] * n
            success = True
            for i in range(n):
                if time.time() > deadline:
                    success = False
                    break
                # Get valid colors for position i
                valid_colors = []
                for c in range(k):
                    # Check all APs ending at i
                    ok = True
                    for a, d in aps_ending[i]:
                        vals = {colors[a], colors[a+d], colors[a+2*d], c}
                        if len(vals) < 3:
                            ok = False
                            break
                    if ok:
                        valid_colors.append(c)
                
                if not valid_colors:
                    success = False
                    break
                else:
                    # Pick a random valid color
                    colors[i] = random.choice(valid_colors)
            
            if success:
                # Final verification
                if is_valid(colors):
                    return colors
        return None

    # Try 3 colors
    result = try_greedy(3)
    if result is not None:
        return result

    # Try 4 colors
    result = try_greedy(4)
    if result is not None:
        return result

    # Try 5 colors
    result = try_greedy(5)
    if result is not None:
        return result

    # Fallback: Return a pattern with many colors, e.g. all distinct?
    # If we use 128 colors, every AP has 4 distinct colors, so it's valid.
    # Value = 128. Score = 1/128.
    # Better than 0.
    return list(range(n))
