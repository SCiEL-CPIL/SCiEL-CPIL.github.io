import time
import random
import math

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Precompute for each index, the list of (o1, o2, o3)
    # where o1, o2, o3 are the other indices in the AP
    aps_with_idx = [[] for _ in range(N)]
    for ap in aps:
        i0, i1, i2, i3 = ap
        aps_with_idx[i0].append((i1, i2, i3))
        aps_with_idx[i1].append((i0, i2, i3))
        aps_with_idx[i2].append((i0, i1, i3))
        aps_with_idx[i3].append((i0, i1, i2))

    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    best_score = float('inf')
    best_colors = None
    
    while time.time() < deadline:
        colors = [random.randint(0, 1) for _ in range(N)]
        current_score = count_mono(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            
        T = 1.0
        cooling = 0.9999
        min_T = 1e-5
        last_improvement_time = time.time()
        
        while time.time() < deadline:
            i = random.randint(0, N - 1)
            
            delta = 0
            for o1, o2, o3 in aps_with_idx[i]:
                c_o1 = colors[o1]
                c_o2 = colors[o2]
                c_o3 = colors[o3]
                
                if c_o1 == c_o2 == c_o3:
                    X = c_o1
                    if colors[i] == X:
                        delta -= 1
                    else:
                        delta += 1
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    last_improvement_time = time.time()
            elif delta == 0:
                if random.random() < 0.5:
                    colors[i] = 1 - colors[i]
            else:
                if random.random() < math.exp(-delta / T):
                    colors[i] = 1 - colors[i]
                    current_score += delta
                    
            T *= cooling
            if T < min_T:
                T = min_T
            
            if time.time() - last_improvement_time > 10.0:
                break

    return best_colors
