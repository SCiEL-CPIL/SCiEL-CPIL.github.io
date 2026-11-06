import time
import numpy as np

def construct():
    n = 256
    # Safety margin: 150s limit, use 140s to be safe
    deadline = time.time() + 140.0

    # 1. Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256
    # Indices 0 to 255
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    if num_aps == 0:
        return [0] * n
        
    aps_arr = np.array(aps, dtype=np.int32)
    
    def get_score(color):
        """
        Vectorized count of monochromatic 4-APs.
        color: numpy array of 0s and 1s of length n.
        """
        # Extract colors for all APs
        # shape: (num_aps, 4)
        c = color[aps_arr]
        # Check if all 4 are equal
        # c[:, 0] == c[:, 1] == c[:, 2] == c[:, 3]
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))

    def local_search(color, time_limit):
        """
        Single-flip local search.
        Returns improved color and score.
        """
        start = time.time()
        current_score = get_score(color)
        improved = True
        passes = 0
        max_passes = 100
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for pos in range(n):
                if time.time() - start > time_limit:
                    return color, current_score
                
                # Try flipping pos
                old_val = color[pos]
                color[pos] = 1 - old_val
                
                new_score = get_score(color)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    color[pos] = old_val
                    
        return color, current_score

    def simulated_annealing_block(color, time_limit, T_start=10.0, T_end=0.1):
        """
        Simulated Annealing with block flips.
        """
        start = time.time()
        current_score = get_score(color)
        best_score = current_score
        best_color = color.copy()
        
        rng = np.random.default_rng()
        T = T_start
        
        # Estimate iterations
        # We will just run until time limit
        while time.time() - start < time_limit:
            # Choose a block to flip
            # Random start and length
            # Lengths: 1, 2, 4, 8, 16, 32
            # Weights favor smaller blocks initially, maybe larger later? 
            # Let's just pick random length from a set.
            block_len = rng.choice([1, 2, 4, 8, 16], p=[0.3, 0.2, 0.2, 0.2, 0.1])
            
            max_start = n - block_len
            if max_start <= 0:
                block_len = 1
                max_start = n - 1
                
            start_idx = rng.integers(0, max_start + 1)
            
            # Create candidate
            candidate = color.copy()
            candidate[start_idx : start_idx + block_len] = 1 - candidate[start_idx : start_idx + block_len]
            
            new_score = get_score(candidate)
            
            if new_score <= current_score:
                color = candidate
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
            else:
                # Accept with probability
                delta = new_score - current_score
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0
                    
                if rng.random() < prob:
                    color = candidate
                    current_score = new_score
            
            # Cool down
            # Linear cooling or exponential? Exponential is standard.
            # We don't know total iterations, so we can use a fixed cooling rate per iteration
            # or cool based on time.
            # Let's use a simple linear cooling towards T_end over the duration.
            elapsed = time.time() - start
            total_time = time_limit
            if elapsed < total_time:
                T = T_start - (T_start - T_end) * (elapsed / total_time)
            else:
                T = T_end
                if T < 0: T = 0

        return best_color, best_score

    # --- Main Execution ---
    
    # Initialize with Thue-Morse sequence
    # Thue-Morse: t(n) = popcount(n) % 2
    thue_morse = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    best_color = thue_morse.copy()
    best_score = get_score(best_color)
    
    # Phase 1: Local Search from Thue-Morse
    t1 = time.time()
    ls_time = min(10.0, deadline - t1 - 20.0)
    if ls_time > 0:
        c, s = local_search(best_color.copy(), ls_time)
        if s < best_score:
            best_score = s
            best_color = c

    # Phase 2: Simulated Annealing with Block Flips
    t2 = time.time()
    sa_time = min(80.0, deadline - t2 - 10.0)
    if sa_time > 0:
        # Run SA from the current best
        c, s = simulated_annealing_block(best_color.copy(), sa_time)
        if s < best_score:
            best_score = s
            best_color = c
            
        # Run SA from a slightly perturbed Thue-Morse
        t3 = time.time()
        remaining_sa = deadline - t3 - 10.0
        if remaining_sa > 10:
            rng = np.random.default_rng(123)
            perturbed = thue_morse.copy()
            # Flip ~5% of bits
            flip_mask = rng.random(n) < 0.05
            perturbed[flip_mask] = 1 - perturbed[flip_mask]
            
            c, s = simulated_annealing_block(perturbed, min(30.0, remaining_sa))
            if s < best_score:
                best_score = s
                best_color = c

    # Phase 3: Final Local Search
    t4 = time.time()
    ls_time_final = deadline - t4 - 2.0
    if ls_time_final > 0:
        c, s = local_search(best_color.copy(), ls_time_final)
        if s < best_score:
            best_score = s
            best_color = c

    # Ensure output is a list of ints
    return [int(x) for x in best_color]
