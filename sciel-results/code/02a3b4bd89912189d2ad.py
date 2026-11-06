import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Using 0-indexed positions
    aps = []
    for d in range(1, N + 1):
        for a in range(1, N - 3 * d + 1):
            # 0-indexed: a-1, a+d-1, a+2d-1, a+3d-1
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Convert to numpy arrays for vectorized operations
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # Build adjacency: for each position i, list of (ap_index, other_positions)
    # This allows fast delta computation when flipping bit i
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d_pos) in enumerate(aps):
        for pos in (a, b, c, d_pos):
            pos_to_aps[pos].append(idx)
    
    def count_mono(coloring):
        """Count monochromatic 4-term APs."""
        # coloring is a numpy array of 0s and 1s
        # For each AP, check if all 4 positions have the same color
        # mono if all equal coloring[0]
        c0 = coloring[aps_arr[:, 0]]
        c1 = coloring[aps_arr[:, 1]]
        c2 = coloring[aps_arr[:, 2]]
        c3 = coloring[aps_arr[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))
    
    def delta_for_flip(coloring, pos):
        """Compute change in objective if we flip bit at position pos."""
        # Before flip: count mono APs containing pos
        # After flip: recount
        # Efficient: for each AP containing pos, check if it's mono before/after
        current_color = coloring[pos]
        delta = 0
        for ap_idx in pos_to_aps[pos]:
            a, b, c, d_pos = aps[ap_idx]
            # Get colors of all 4 positions
            colors = [coloring[a], coloring[b], coloring[c], coloring[d_pos]]
            # Is it mono before?
            is_mono_before = (colors[0] == colors[1] == colors[2] == colors[3])
            # Flip the color at pos
            flip_idx = [a, b, c, d_pos].index(pos)
            colors[flip_idx] = 1 - colors[flip_idx]
            is_mono_after = (colors[0] == colors[1] == colors[2] == colors[3])
            if is_mono_before and not is_mono_after:
                delta -= 1
            elif not is_mono_before and is_mono_after:
                delta += 1
        return delta
    
    # Initial coloring: Thue-Morse sequence (known to be good for avoiding APs)
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        tm = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    # Try multiple initializations and pick the best
    initializations = []
    
    # Thue-Morse
    initializations.append(thue_morse(N))
    
    # Thue-Morse flipped
    initializations.append(1 - thue_morse(N))
    
    # Random initializations
    random.seed(42)
    for _ in range(5):
        initializations.append(np.random.randint(0, 2, N, dtype=np.int8))
    
    # Block-based initialization (like seed but varied)
    for block_size in [2, 3, 4, 5, 6, 8]:
        init = np.zeros(N, dtype=np.int8)
        for i in range(N):
            init[i] = (i // block_size) % 2
        initializations.append(init)
    
    # Find best initial coloring
    best_coloring = None
    best_score = float('inf')
    for init in initializations:
        score = count_mono(init)
        if score < best_score:
            best_score = score
            best_coloring = init.copy()
    
    # Simulated Annealing with multiple restarts
    def simulated_annealing(initial, time_limit):
        coloring = initial.copy()
        current_score = count_mono(coloring)
        best_coloring = coloring.copy()
        best_score = current_score
        
        # SA parameters
        T_start = 5.0
        T_end = 0.01
        alpha = 0.9995  # Cooling rate
        
        T = T_start
        no_improve = 0
        max_no_improve = 20000
        
        while T > T_end:
            # Check time
            if time.time() > time_limit:
                break
            
            # Random position to flip
            pos = random.randint(0, N - 1)
            delta = delta_for_flip(coloring, pos)
            
            # Accept if improvement, or with probability exp(-delta/T) if worsening
            if delta <= 0:
                coloring[pos] = 1 - coloring[pos]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Probabilistic acceptance
                if random.random() < np.exp(-delta / T):
                    coloring[pos] = 1 - coloring[pos]
                    current_score += delta
                    no_improve += 1
                else:
                    no_improve += 1
            
            T *= alpha
            
            # Restart if stuck
            if no_improve > max_no_improve:
                # Perturb from best
                for _ in range(10):
                    p = random.randint(0, N - 1)
                    coloring[p] = 1 - coloring[p]
                current_score = count_mono(coloring)
                T = T_start * 0.5
                no_improve = 0
        
        return best_coloring, best_score
    
    # Run SA with multiple restarts
    best_result = (best_coloring, best_score)
    
    # First SA run from best initial
    if time.time() < deadline - 10:
        result, score = simulated_annealing(best_coloring, deadline - 5)
        if score < best_result[1]:
            best_result = result
    
    # Multiple restarts with perturbations of global best
    restarts = 0
    while time.time() < deadline - 5:
        # Perturb the best coloring
        perturbed = best_result[0].copy()
        num_flips = random.randint(5, 20)
        for _ in range(num_flips):
            p = random.randint(0, N - 1)
            perturbed[p] = 1 - perturbed[p]
        
        result, score = simulated_annealing(perturbed, deadline - 5)
        if score < best_result[1]:
            best_result = (result, score)
        restarts += 1
    
    # Final local search polish: try all single flips and accept improvements
    coloring = best_result[0].copy()
    score = best_result[1]
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for pos in range(N):
            if time.time() > deadline - 2:
                break
            delta = delta_for_flip(coloring, pos)
            if delta < 0:
                coloring[pos] = 1 - coloring[pos]
                score += delta
                improved = True
    
    # Return as list of ints
    return [int(c) for c in coloring]
