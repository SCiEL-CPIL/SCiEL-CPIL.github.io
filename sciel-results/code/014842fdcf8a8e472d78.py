    import numpy as np
    import time
    import random
    import math

    def construct():
        N = 256
        start_time = time.time()
        deadline = start_time + 110.0
        
        # Precomputation
        aps = []
        for a in range(1, N + 1):
            for d in range(1, N + 1):
                if a + 3 * d <= N:
                    aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
        
        num_aps = len(aps)
        pos_to_aps = [[] for _ in range(N)]
        ap_other_indices = []
        ap_pos_map = [] # ap_pos_map[ap_idx] is a dict or list mapping index to position k
        
        for ap_idx, ap in enumerate(aps):
            others = [None] * 4
            pos_map = {}
            for k in range(4):
                others[k] = tuple(ap[j] for j in range(4) if j != k)
                pos_to_aps[ap[k]].append(ap_idx)
                pos_map[ap[k]] = k
            ap_other_indices.append(others)
            ap_pos_map.append(pos_map)
            
        def count_mono(c):
            count = 0
            for ap in aps:
                if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                    count += 1
            return count

        # Initialize
        colors = np.random.randint(0, 2, N)
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = 50.0
        cooling = 0.999993
        steps_since_improvement = 0
        max_steps_without_improvement = 10000
        
        while time.time() < deadline:
            i = random.randint(0, N - 1)
            delta = 0
            c_i = colors[i]
            
            for ap_idx in pos_to_aps[i]:
                k = ap_pos_map[ap_idx][i]
                others = ap_other_indices[ap_idx][k]
                
                c_o1 = colors[others[0]]
                c_o2 = colors[others[1]]
                c_o3 = colors[others[2]]
                
                if c_o1 == c_o2 == c_o3:
                    if c_i == c_o1:
                        delta -= 1
                    else:
                        delta += 1
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[i] = 1 - colors[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                    
                if steps_since_improvement > max_steps_without_improvement:
                    T = 50.0
                    steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement > max_steps_without_improvement:
                    T = 50.0
                    steps_since_improvement = 0
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return list(best_colors)
    