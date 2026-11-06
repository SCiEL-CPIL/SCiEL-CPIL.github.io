import random
import time

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 55
    
    # Generate all non-constant 4-term progressions
    aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                aps.append((a - 1, b - 1, c - 1, e - 1))
                
    num_aps = len(aps)
    
    def count_violations(colors):
        v = 0
        for a, b, c, e in aps:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                v += 1
        return v

    # Try to find a 3-coloring
    # Use multiple restarts
    for restart in range(50):
        current_colors = [random.randint(0, 2) for _ in range(N)]
        current_violations = count_violations(current_colors)
        
        if current_violations == 0:
            return current_colors

        T = 1.0
        cooling = 0.9995
        no_improve_count = 0
        best_in_run = current_violations
        best_colors_in_run = current_colors[:]

        while time.time() < deadline - 5:
            # Pick random AP
            idx = random.randint(0, num_aps - 1)
            a, b, c, e = aps[idx]
            ca, cb, cc, ce = current_colors[a], current_colors[b], current_colors[c], current_colors[e]
            
            if len({ca, cb, cc, ce}) >= 3:
                continue

            # Pick random position
            pos_idx = random.randint(0, 3)
            if pos_idx == 0: pos = a
            elif pos_idx == 1: pos = b
            elif pos_idx == 2: pos = c
            else: pos = e
            
            old_color = current_colors[pos]
            new_color = random.choice([c for c in range(3) if c != old_color])
            
            current_colors[pos] = new_color
            new_violations = count_violations(current_colors)
            
            if new_violations < current_violations:
                current_violations = new_violations
                no_improve_count = 0
                if new_violations < best_in_run:
                    best_in_run = new_violations
                    best_colors_in_run = current_colors[:]
                if new_violations == 0:
                    return current_colors
            else:
                delta = new_violations - current_violations
                # Accept with probability
                if random.random() < pow(2.71828, -delta / T):
                    current_violations = new_violations
                    no_improve_count += 1
                else:
                    current_colors[pos] = old_color
                    no_improve_count += 1
                    
            T *= cooling
            if T < 0.01:
                break
            
            # If stuck for too long, break inner loop and restart
            if no_improve_count > 1000:
                break
        
        # End of one restart
        if best_in_run == 0:
            return best_colors_in_run

    # Fallback: Greedy
    colors = []
    for x in range(1, N + 1):
        found = False
        for color in range(128):
            valid = True
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                i1 = x - 1 - 3 * d
                i2 = x - 1 - 2 * d
                i3 = x - 1 - d
                if i1 < 0: continue
                c1 = colors[i1]
                c2 = colors[i2]
                c3 = colors[i3]
                if len({c1, c2, c3, color}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
            
    return colors
