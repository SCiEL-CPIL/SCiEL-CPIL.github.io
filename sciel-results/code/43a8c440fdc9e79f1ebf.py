import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # For each position, list of (ap_index, offset)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Precompute APs as numpy array for fast counting
    aps_arr = np.array(aps, dtype=np.int32)
    
    def count_mono(color):
        """Count monochromatic 4-APs using numpy."""
        c = np.array(color, dtype=np.int8)
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def delta_flip(color, mono, p):
        """Compute delta in mono count when flipping position p."""
        delta = 0
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            if mono[ap_idx]:
                delta -= 1
            else:
                new_val = 1 - color[p]
                others = []
                for j, pos in enumerate((p0, p1, p2, p3)):
                    if j != offset:
                        others.append(color[pos])
                if all(v == new_val for v in others):
                    delta += 1
        return delta
    
    def local_search(color, max_sweeps=300):
        color = list(color)
        current = count_mono(color)
        mono = [False] * num_aps
        for idx, (p0, p1, p2, p3) in enumerate(aps):
            if color[p0] == color[p1] == color[p2] == color[p3]:
                mono[idx] = True
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 3:
                break
            
            # Try all positions, find best flip
            best_delta = 0
            best_pos = -1
            for p in range(n):
                d = delta_flip(color, mono, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            
            if best_pos < 0:
                break
            
            color[best_pos] = 1 - color[best_pos]
            
            # Update mono status for affected APs
            for ap_idx, offset in pos_aps[best_pos]:
                p0, p1, p2, p3 = aps[ap_idx]
                was_mono = mono[ap_idx]
                mono[ap_idx] = (color[p0] == color[p1] == color[p2] == color[p3])
                if mono[ap_idx] and not was_mono:
                    current += 1
                elif not mono[ap_idx] and was_mono:
                    current -= 1
        
        return color, current
    
    # Thue-Morse initialization
    color = [bin(i).count('1') % 2 for i in range(n)]
    
    # Initial local search
    color, score = local_search(color)
    
    best_color = color[:]
    best_score = score
    
    # Random restarts with varying perturbation
    random.seed(12345)
    while time.time() < deadline - 5:
        # Vary the number of flips for diversity
        n_flips = random.randint(3, 30)
        trial = best_color[:]
        for _ in range(n_flips):
            p = random.randint(0, n - 1)
            trial[p] = 1 - trial[p]
        
        trial, s = local_search(trial, max_sweeps=200)
        if s < best_score:
            best_score = s
            best_color = trial[:]
    
    return best_color
