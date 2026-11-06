import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Using 0-indexed: positions 0..N-1
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # NumPy array for vectorized full evaluation
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, list of AP indices containing it
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def count_mono_np(colors):
        """Count monochromatic 4-term APs using NumPy vectorization."""
        # colors is a numpy array of 0/1
        c0 = colors[ap_indices[:, 0]]
        c1 = colors[ap_indices[:, 1]]
        c2 = colors[ap_indices[:, 2]]
        c3 = colors[ap_indices[:, 3]]
        # All four equal: (c0==c1) & (c1==c2) & (c2==c3)
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())
    
    def count_mono(colors):
        """Count monochromatic 4-term APs."""
        return count_mono_np(colors)
    
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
    
    # Random initializations (more of them)
    for _ in range(8):
        initializations.append(np.random.randint(0, 2, N, dtype=np.int8))
    
    # Alternating blocks of various sizes
    for block in [2, 3, 4, 5, 7, 8]:
        initializations.append(np.array([(i // block) % 2 for i in range(N)], dtype=np.int8))
    
    # Van der Corput / low-discrepancy style
    for period in [16, 32, 64]:
        pattern = np.array([((i * 7) % period) < period // 2 for i in range(N)], dtype=np.int8)
        initializations.append(pattern)
    
    for init_colors in initializations:
        if time.time() > deadline - 15:
            break
        
        colors = init_colors.copy()
        current_count = count_mono(colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
        
        # Local search with simulated annealing
        T = 3.0
        cooling = 0.9995
        min_T = 0.005
        no_improve = 0
        max_no_improve = 8000
        
        while T > min_T and time.time() < deadline - 5:
            pos = random.randint(0, N - 1)
            d = delta_for_flip(colors, pos)
            
            if d < 0 or (d == 0 and random.random() < 0.15) or (d > 0 and random.random() < np.exp(-d / T)):
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
                # Restart with perturbation: flip a random subset
                n_perturb = random.randint(5, 20)
                perturb_positions = random.sample(range(N), n_perturb)
                for p in perturb_positions:
                    colors[p] = 1 - colors[p]
                current_count = count_mono(colors)
                T = max(T, 2.0)
                no_improve = 0
            
            T *= cooling
        
        # Final greedy pass: multiple sweeps
        for sweep in range(3):
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                # Shuffle order for diversity
                order = list(range(N))
                random.shuffle(order)
                for pos in order:
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
    
    # Final high-precision polish: try all single flips on best solution
    if best_colors is not None and time.time() < deadline - 1:
        colors = best_colors.copy()
        current_count = best_count
        for pos in range(N):
            d = delta_for_flip(colors, pos)
            if d < 0:
                colors[pos] = 1 - colors[pos]
                current_count += d
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
        # Second pass
        for pos in range(N):
            d = delta_for_flip(colors, pos)
            if d < 0:
                colors[pos] = 1 - colors[pos]
                current_count += d
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
    
    # Convert to list of ints 0/1
    result = [int(c) for c in best_colors]
    return result
