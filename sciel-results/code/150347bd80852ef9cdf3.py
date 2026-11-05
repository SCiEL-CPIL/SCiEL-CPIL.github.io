import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    
    # Precompute APs
    aps = []
    for d in range(1, 86):
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    aps = np.array(aps, dtype=np.int32)
    n_aps = len(aps)
    
    # Initial random coloring
    random.seed(42)
    colors = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
    
    def count_mono(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())
    
    current_count = count_mono(colors)
    best_colors = colors.copy()
    best_count = current_count
    
    def local_search(colors, current_count):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            deltas = np.zeros(256, dtype=np.int32)
            
            c0 = colors[aps[:, 0]]
            c1 = colors[aps[:, 1]]
            c2 = colors[aps[:, 2]]
            c3 = colors[aps[:, 3]]
            
            # Loop over APs to compute deltas
            for j in range(n_aps):
                v0 = c0[j]
                v1 = c1[j]
                v2 = c2[j]
                v3 = c3[j]
                
                # Check if 3 are equal
                if v1 == v2 == v3:
                    v = v1
                    ci = v0
                    pos = aps[j, 0]
                    deltas[pos] += -1 if ci == v else 1
                if v0 == v2 == v3:
                    v = v0
                    ci = v1
                    pos = aps[j, 1]
                    deltas[pos] += -1 if ci == v else 1
                if v0 == v1 == v3:
                    v = v0
                    ci = v2
                    pos = aps[j, 2]
                    deltas[pos] += -1 if ci == v else 1
                if v0 == v1 == v2:
                    v = v0
                    ci = v3
                    pos = aps[j, 3]
                    deltas[pos] += -1 if ci == v else 1
            
            best_pos = np.argmin(deltas)
            best_delta = deltas[best_pos]
            
            if best_delta < 0:
                colors[best_pos] = 1 - colors[best_pos]
                current_count += best_delta
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
        
        return colors, current_count
    
    # ILS Loop
    colors, current_count = local_search(colors, current_count)
    
    while time.time() < deadline - 10:
        # Perturb
        k = random.randint(5, 20)
        for _ in range(k):
            pos = random.randint(0, 255)
            colors[pos] = 1 - colors[pos]
        
        # Local Search
        colors, current_count = local_search(colors, current_count)
        
        # If stuck, try random restart
        if current_count >= best_count and random.random() < 0.1:
             colors = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
             current_count = count_mono(colors)
    
    return [int(c) for c in best_colors]
