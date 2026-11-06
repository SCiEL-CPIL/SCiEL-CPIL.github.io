import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128
    
    # Precompute all nonconstant 4-term progressions (0-indexed)
    progressions = []
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i3 = a + 3 * d
            if i3 < N:
                progressions.append((a, a + d, a + 2 * d, i3))
    
    num_progs = len(progressions)
    
    # For each position, list of progression indices containing it
    pos_to_progs = [[] for _ in range(N)]
    for idx, prog in enumerate(progressions):
        for p in prog:
            pos_to_progs[p].append(idx)
    
    def count_violations(colors):
        """Count number of 4-term progressions with < 3 distinct colors."""
        count = 0
        for prog in progressions:
            i0, i1, i2, i3 = prog
            if len({colors[i0], colors[i1], colors[i2], colors[i3]}) < 3:
                count += 1
        return count
    
    def is_valid(colors):
        """Check if coloring is valid."""
        for prog in progressions:
            i0, i1, i2, i3 = prog
            if len({colors[i0], colors[i1], colors[i2], colors[i3]}) < 3:
                return False
        return True
    
    # Try k = 3, 4, 5 with random restart local search
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5.0:
            break
        
        best_colors = None
        best_viol = float('inf')
        
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 3.0:
                break
            
            # Random initialization
            colors = [random.randint(0, k - 1) for _ in range(N)]
            current_viol = count_violations(colors)
            
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
            
            if current_viol == 0:
                break
            
            # Local search
            for iteration in range(3000):
                if time.time() > deadline - 2.0:
                    break
                
                if current_viol == 0:
                    break
                
                # Pick a random position and try to improve by changing its color
                pos = random.randint(0, N - 1)
                old_color = colors[pos]
                
                # Try each alternative color
                best_new_color = None
                best_new_viol = current_viol
                
                for c in range(k):
                    if c == old_color:
                        continue
                    colors[pos] = c
                    # Only need to check progressions containing pos
                    new_viol = 0
                    for pj in pos_to_progs[pos]:
                        prog = progressions[pj]
                        if len({colors[p] for p in prog}) < 3:
                            new_viol += 1
                    colors[pos] = old_color
                    
                    if new_viol < best_new_viol:
                        best_new_viol = new_viol
                        best_new_color = c
                
                if best_new_color is not None:
                    colors[pos] = best_new_color
                    current_viol = best_new_viol
                    if current_viol < best_viol:
                        best_viol = current_viol
                        best_colors = colors[:]
                else:
                    # No improvement; try a random perturbation
                    if random.random() < 0.1:
                        new_c = random.randint(0, k - 1)
                        colors[pos] = new_c
                        # Recompute violations for affected progs
                        current_viol = count_violations(colors)
                        if current_viol < best_viol:
                            best_viol = current_viol
                            best_colors = colors[:]
            
            if best_viol == 0:
                break
        
        if best_colors is not None and best_viol == 0:
            if is_valid(best_colors):
                return best_colors
    
    # Fallback: use greedy seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    continue
                s = {color}
                for j in (1, 2, 3):
                    idx = x - 1 - j * d
                    if 0 <= idx < len(colors):
                        s.add(colors[idx])
                if len(s) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
        else:
            colors.append(0)
    
    return colors
