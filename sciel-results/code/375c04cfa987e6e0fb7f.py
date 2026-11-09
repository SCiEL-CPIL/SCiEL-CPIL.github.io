import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 128
    best_colors = None
    best_count = 128  # Max possible distinct colors
    
    def is_valid_partial(colors, idx, color):
        max_d = idx // 3
        for d in range(1, max_d + 1):
            i1 = idx - d
            i2 = idx - 2 * d
            i3 = idx - 3 * d
            
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]
            
            if len({color, c1, c2, c3}) < 3:
                return False
        return True

    def is_valid_full(colors):
        n = len(colors)
        for x in range(1, n + 1):
            max_d = (n - x) // 3
            for d in range(1, max_d + 1):
                i0 = x - 1
                i1 = x + d - 1
                i2 = x + 2 * d - 1
                i3 = x + 3 * d - 1
                
                c0 = colors[i0]
                c1 = colors[i1]
                c2 = colors[i2]
                c3 = colors[i3]
                
                if len({c0, c1, c2, c3}) < 3:
                    return False
        return True

    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        colors = [None] * n
        used_colors = set()
        
        success = True
        for i in range(n):
            if time.time() > deadline:
                success = False
                break
                
            valid_colors = []
            if used_colors:
                shuffled_used = list(used_colors)
                random.shuffle(shuffled_used)
                for c in shuffled_used:
                    if is_valid_partial(colors, i, c):
                        valid_colors.append(c)
            
            if not valid_colors:
                for c in range(128):
                    if c in used_colors:
                        continue
                    if is_valid_partial(colors, i, c):
                        valid_colors.append(c)
                        break
            
            if not valid_colors:
                success = False
                break
            
            chosen = random.choice(valid_colors)
            colors[i] = chosen
            used_colors.add(chosen)
            
        if not success:
            continue
            
        if is_valid_full(colors):
            count = len(used_colors)
            if count < best_count:
                best_count = count
                best_colors = colors[:]
                if best_count == 3:
                    break

    if best_colors is None:
        # Fallback to deterministic greedy
        colors = [None] * n
        used_colors = set()
        for i in range(n):
            valid_colors = []
            if used_colors:
                for c in sorted(used_colors):
                    if is_valid_partial(colors, i, c):
                        valid_colors.append(c)
            if not valid_colors:
                for c in range(128):
                    if c in used_colors:
                        continue
                    if is_valid_partial(colors, i, c):
                        valid_colors.append(c)
                        break
            if not valid_colors:
                return [0] * n 
            colors[i] = valid_colors[0]
            used_colors.add(colors[i])
        
        if is_valid_full(colors):
            return colors
        else:
            return colors

    return best_colors
