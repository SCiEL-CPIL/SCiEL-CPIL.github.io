import random
import time

def check_validity(colors):
    n = len(colors)
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            c1 = colors[i]
            c2 = colors[i + d]
            c3 = colors[i + 2 * d]
            c4 = colors[i + 3 * d]
            if len({c1, c2, c3, c4}) < 3:
                return False
    return True

def count_violations(colors):
    n = len(colors)
    violations = 0
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            c1 = colors[i]
            c2 = colors[i + d]
            c3 = colors[i + 2 * d]
            c4 = colors[i + 3 * d]
            if len({c1, c2, c3, c4}) < 3:
                violations += 1
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_colors = None
    best_distinct = 129
    
    # Try increasing number of colors
    for max_colors in range(3, 10):
        if time.time() > deadline:
            break
            
        # Multiple restarts
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Initialize with random colors from 0 to max_colors-1
            colors = [random.randint(0, max_colors - 1) for _ in range(128)]
            
            # Local Search
            for step in range(5000):
                if time.time() > deadline:
                    break
                    
                violations = count_violations(colors)
                if violations == 0:
                    distinct = len(set(colors))
                    if distinct < best_distinct:
                        best_distinct = distinct
                        best_colors = colors.copy()
                    # Try to reduce colors further?
                    # For now, accept and break to try smaller max_colors or better distinct
                    break
                
                # Pick a random violation
                # To make it faster, we can just pick a random index and step and check
                # But finding a specific violation is better.
                # Let's just pick a random position and try to change it
                # If it's part of a violation, it might help.
                
                idx = random.randint(0, 127)
                old_color = colors[idx]
                
                # Try changing to a different color
                # Prefer colors already in use to keep distinct count low
                used_colors = list(set(colors))
                random.shuffle(used_colors)
                
                improved = False
                for new_color in used_colors:
                    if new_color == old_color:
                        continue
                    colors[idx] = new_color
                    new_violations = count_violations(colors)
                    if new_violations < violations:
                        improved = True
                        break
                    else:
                        colors[idx] = old_color
                
                if not improved:
                    # Try introducing a new color if allowed
                    if len(used_colors) < max_colors:
                        new_color = max_colors - 1 # Or random new one
                        # Actually, just pick a random color from 0 to max_colors-1
                        new_color = random.randint(0, max_colors - 1)
                        if new_color != old_color:
                            colors[idx] = new_color
                            new_violations = count_violations(colors)
                            if new_violations < violations:
                                improved = True
                            else:
                                colors[idx] = old_color
                    if not improved:
                        # Stuck, restart or break
                        # For now, just break to restart
                        break
            
            # If we found a valid one with fewer distinct colors, update best
            if check_validity(colors):
                distinct = len(set(colors))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = colors.copy()
    
    if best_colors:
        # Ensure labels are in 0..127 (they are)
        # Return the best found
        return best_colors
    else:
        # Fallback: Greedy (Seed logic)
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
                    and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))):
                    colors.append(color)
                    break
        return colors
