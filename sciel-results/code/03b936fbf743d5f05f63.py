import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 140.0  # 10s margin before 150s kill
    
    # Thue-Morse initialization
    colors = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def count_mono():
        count = 0
        for d in range(1, (n - 3) // 3 + 1):
            for a in range(n - 3 * d):
                if colors[a] == colors[a+d] == colors[a+2*d] == colors[a+3*d]:
                    count += 1
        return count

    current_cost = count_mono()
    best_cost = current_cost
    best_colors = colors.copy()
    
    max_d = (n - 3) // 3
    
    # Precompute for each position x, the list of (d, k, a) where x = a + k*d
    # and a >= 0, a + 3*d < n
    pos_aps = [[] for _ in range(n)]
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            for k in range(4):
                x = a + k * d
                pos_aps[x].append((d, k, a))
    
    # Convert to numpy arrays for faster access
    pos_aps_arr = []
    for x in range(n):
        if pos_aps[x]:
            arr = np.array(pos_aps[x], dtype=np.int32)  # columns: d, k, a
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 3), dtype=np.int32))
    
    def delta_flip(x):
        """Calculate change in cost if we flip color at x."""
        aps = pos_aps_arr[x]
        if len(aps) == 0:
            return 0
        
        d_arr = aps[:, 0]
        k_arr = aps[:, 1]
        a_arr = aps[:, 2]
        
        # Get indices for the 4 positions in each AP
        idx0 = a_arr
        idx1 = a_arr + d_arr
        idx2 = a_arr + 2 * d_arr
        idx3 = a_arr + 3 * d_arr
        
        v0 = colors[idx0]
        v1 = colors[idx1]
        v2 = colors[idx2]
        v3 = colors[idx3]
        
        old_mono = (v0 == v1) & (v1 == v2) & (v2 == v3)
        
        # Flip x: determine which index position x corresponds to
        nv0 = v0.copy()
        nv1 = v1.copy()
        nv2 = v2.copy()
        nv3 = v3.copy()
        
        mask0 = (k_arr == 0)
        mask1 = (k_arr == 1)
        mask2 = (k_arr == 2)
        mask3 = (k_arr == 3)
        
        nv0[mask0] = 1 - nv0[mask0]
        nv1[mask1] = 1 - nv1[mask1]
        nv2[mask2] = 1 - nv2[mask2]
        nv3[mask3] = 1 - nv3[mask3]
        
        new_mono = (nv0 == nv1) & (nv1 == nv2) & (nv2 == nv3)
        
        delta = int(np.sum(new_mono.astype(np.int8)) - np.sum(old_mono.astype(np.int8)))
        return delta

    # Simulated Annealing
    T = 1.0
    cool_rate = 0.995
    min_T = 0.005
    steps = 0
    
    while time.time() < deadline - 10:
        x = random.randint(0, n - 1)
        d_cost = delta_flip(x)
        
        if d_cost < 0:
            colors[x] = 1 - colors[x]
            current_cost += d_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        else:
            if T > min_T and d_cost > 0:
                p = np.exp(-d_cost / T)
                if random.random() < p:
                    colors[x] = 1 - colors[x]
                    current_cost += d_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
        
        T *= cool_rate
        steps += 1
    
    # Final polish: local search with only improving moves
    polish_deadline = deadline
    for _ in range(50):  # Up to 50 passes of local search
        if time.time() > polish_deadline:
            break
        improved = False
        for x in range(n):
            if time.time() > polish_deadline:
                break
            d_cost = delta_flip(x)
            if d_cost < 0:
                colors[x] = 1 - colors[x]
                current_cost += d_cost
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
        if not improved:
            break
    
    return best_colors.tolist()
