import random
import math
import time

def construct():
    N = 256
    # Generate all 4-APs and map indices to their AP neighbors
    neighbors = [[] for _ in range(N)]
    ap_list = []
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            ap = (i1, i2, i3, i4)
            ap_list.append(ap)
            neighbors[i1].append((i2, i3, i4))
            neighbors[i2].append((i1, i3, i4))
            neighbors[i3].append((i1, i2, i4))
            neighbors[i4].append((i1, i2, i3))

    def calculate_score(colors):
        count = 0
        for ap in ap_list:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c1 == c2 == c3 == c4:
                count += 1
        return count

    def simulated_annealing(initial_colors, time_limit=10.0):
        colors = initial_colors[:]
        current_score = calculate_score(colors)
        best_score = current_score
        best_colors = colors[:]
        
        start_time = time.time()
        T = 1.0
        alpha = 0.99999
        min_T = 0.001
        
        while T > min_T and (time.time() - start_time) < time_limit:
            i = random.randint(0, N - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta_e = 0
            for j, k, l in neighbors[i]:
                cj, ck, cl = colors[j], colors[k], colors[l]
                if cj == ck == cl:
                    if old_color == cj:
                        delta_e -= 1
                    else:
                        delta_e += 1
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                colors[i] = new_color
                current_score += delta_e
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            
            T *= alpha
            
        return best_colors, best_score

    best_overall_score = float('inf')
    best_overall_colors = None
    
    global_start = time.time()
    total_time_limit = 110.0
    
    for restart in range(10):
        if time.time() - global_start > total_time_limit:
            break
            
        initial_colors = [random.randint(0, 1) for _ in range(N)]
        
        elapsed = time.time() - global_start
        remaining = total_time_limit - elapsed
        if remaining <= 0:
            break
        
        run_time = min(10.0, remaining)
        
        res_colors, res_score = simulated_annealing(initial_colors, time_limit=run_time)
        
        if res_score < best_overall_score:
            best_overall_score = res_score
            best_overall_colors = res_colors

    if best_overall_colors is None:
        best_overall_colors = [random.randint(0, 1) for _ in range(N)]
        
    return best_overall_colors
