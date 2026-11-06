import random
import time

def construct():
    n = 128
    start_time = time.time()
    time_limit = 55.0

    # Precompute APs
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
            
    num_aps = len(aps)

    def count_bad(colors):
        bad = 0
        for p1, p2, p3, p4 in aps:
            if len({colors[p1], colors[p2], colors[p3], colors[p4]}) < 3:
                bad += 1
        return bad

    # Try k=3, 4, 5
    for k in range(3, 6):
        best_colors = None
        best_bad = float('inf')
        
        # Multiple restarts
        while time.time() - start_time < time_limit - 5.0:
            colors = [random.randint(0, k-1) for _ in range(n)]
            bad = count_bad(colors)
            
            if bad == 0:
                return colors
            
            # Hill Climbing
            for _ in range(1000):
                if time.time() - start_time > time_limit - 5.0:
                    break
                if bad == 0:
                    return colors
                    
                pos = random.randint(0, n-1)
                old_color = colors[pos]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_bad = count_bad(colors)
                
                if new_bad < bad:
                    bad = new_bad
                else:
                    colors[pos] = old_color
                    
            if bad < best_bad:
                best_bad = bad
                best_colors = colors.copy()
                
        if best_bad == 0:
            return best_colors

    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                a = x - 1 - 3*d
                if a < 0: continue
                if len({colors[a], colors[a+d], colors[a+2*d], color}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
