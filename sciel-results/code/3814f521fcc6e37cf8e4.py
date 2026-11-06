import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Using 0-indexed: positions 0..N-1
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # For each position, list of AP indices containing it
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def count_mono(colors):
        """Count monochromatic 4-term APs."""
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    def delta_for_flip(colors, pos):
        """Compute change in mono count if we flip position pos."""
        c = colors[pos]
        nc = 1 - c
        gain = 0  # APs that become mono after flip
        loss = 0  # APs that stop being mono after flip
        for ap_idx in pos_aps[pos]:
            ap = aps[ap_idx]
            others = [ap[j] for j in range(4) if ap[j] != pos]
            oc = colors[others[0]]
            if oc == colors[others[1]] == colors[others[2]]:
                if oc == nc:
                    gain += 1
                else:  # oc == c
                    loss += 1
        return gain - loss
    
    # Thue-Morse coloring
    def thue_morse(n):
        # Count trailing 1s in binary representation of n (or use bit trick)
        x = n
        c = 0
        while x:
            c ^= 1
            x >>= 1
        return c
    
    # Try multiple initial colorings and local search
    best_colors = None
    best_count = float('inf')
    
    initializations = []
    
    # Thue-Morse
    tm = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    initializations.append(tm)
    
    # Random initializations
    for _ in range(5):
        initializations.append(np.random.randint(0, 2, N, dtype=np.int8))
    
    # Alternating blocks of various sizes
    for block in [2, 3, 4, 5]:
        initializations.append(np.array([(i // block) % 2 for i in range(N)], dtype=np.int8))
    
    for init_colors in initializations:
        if time.time() > deadline - 10:
            break
        
        colors = init_colors.copy()
        current_count = count_mono(colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
        
        # Local search with simulated annealing
        T = 2.0
        cooling = 0.999
        min_T = 0.01
        no_improve = 0
        max_no_improve = 5000
        
        while T > min_T and time.time() < deadline - 5:
            pos = random.randint(0, N - 1)
            d = delta_for_flip(colors, pos)
            
            if d < 0 or (d == 0 and random.random() < 0.1) or (d > 0 and random.random() < np.exp(-d / T)):
                colors[pos] = 1 - colors[pos]
                current_count += d
                no_improve += 1
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
                    no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                # Restart with perturbation
                for _ in range(10):
                    p = random.randint(0, N - 1)
                    colors[p] = 1 - colors[p]
                current_count = count_mono(colors)
                T = max(T, 1.0)
                no_improve = 0
            
            T *= cooling
        
        # Final greedy pass
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for pos in range(N):
                d = delta_for_flip(colors, pos)
                if d < 0:
                    colors[pos] = 1 - colors[pos]
                    current_count += d
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors.copy()
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
    
    # Convert to list of ints 0/1
    result = [int(c) for c in best_colors]
    return result
