import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    n = 256
    
    # Precompute all 4-term APs as 0-indexed tuples
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Build per-position list of AP indices and the other 3 positions
    # For position i, store: list of (ap_index, other_pos1, other_pos2, other_pos3)
    pos_aps = [[] for _ in range(n)]
    for j in range(M):
        p0, p1, p2, p3 = aps_arr[j]
        pos_aps[p0].append((j, p1, p2, p3))
        pos_aps[p1].append((j, p0, p2, p3))
        pos_aps[p2].append((j, p0, p1, p3))
        pos_aps[p3].append((j, p0, p1, p2))
    
    # Convert to numpy arrays for each position
    pos_aps_arrays = []
    for i in range(n):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arrays.append(arr)
        else:
            pos_aps_arrays.append(np.empty((0, 4), dtype=np.int32))
    
    def count_mono(colors):
        """Count monochromatic 4-term APs."""
        total = 0
        for j in range(M):
            p0, p1, p2, p3 = aps_arr[j]
            c0, c1, c2, c3 = colors[p0], colors[p1], colors[p2], colors[p3]
            if c0 == c1 == c2 == c3:
                total += 1
        return total
    
    def count_mono_vec(colors):
        """Vectorized count of monochromatic 4-term APs."""
        c = colors[aps_arr]  # shape (M, 4)
        # mono if all 4 are equal
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return np.sum(mono)
    
    def get_delta(colors, i):
        """Compute delta in mono count when flipping position i."""
        arr = pos_aps_arrays[i]
        if len(arr) == 0:
            return 0
        # arr has shape (k, 4): each row is (ap_idx, other1, other2, other3)
        ap_indices = arr[:, 0]
        others = arr[:, 1:]  # shape (k, 3)
        
        # Get current colors of the 3 other positions
        other_colors = colors[others]  # shape (k, 3)
        cur_color = colors[i]
        new_color = 1 - cur_color
        
        # Before flip: AP is mono if all 4 same
        # The 4 colors are: cur_color, other_colors[:,0], other_colors[:,1], other_colors[:,2]
        before_mono = (cur_color == other_colors[:, 0]) & \
                      (cur_color == other_colors[:, 1]) & \
                      (cur_color == other_colors[:, 2])
        
        # After flip: AP is mono if all 4 same with new_color
        after_mono = (new_color == other_colors[:, 0]) & \
                     (new_color == other_colors[:, 1]) & \
                     (new_color == other_colors[:, 2])
        
        delta = np.sum(after_mono) - np.sum(before_mono)
        return int(delta)
    
    def greedy_polish(colors):
        """Flip positions that reduce mono count until no improvement."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                d = get_delta(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    improved = True
        return colors
    
    # Initial colorings to try
    def random_coloring():
        return np.random.randint(0, 2, n)
    
    def thue_morse_coloring():
        # Thue-Morse sequence
        colors = np.zeros(n, dtype=np.int32)
        for i in range(n):
            # Count bits in i
            colors[i] = bin(i).count('1') % 2
        return colors
    
    def block_coloring(block_size):
        colors = np.zeros(n, dtype=np.int32)
        for i in range(n):
            colors[i] = (i // block_size) % 2
        return colors
    
    # Try several initializations
    best_colors = None
    best_count = float('inf')
    
    # Thue-Morse
    colors = thue_morse_coloring()
    cnt = count_mono_vec(colors)
    if cnt < best_count:
        best_count = cnt
        best_colors = colors.copy()
    
    # Block colorings
    for bs in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        colors = block_coloring(bs)
        cnt = count_mono_vec(colors)
        if cnt < best_count:
            best_count = cnt
            best_colors = colors.copy()
    
    # Random initializations
    for _ in range(20):
        colors = random_coloring()
        cnt = count_mono_vec(colors)
        if cnt < best_count:
            best_count = cnt
            best_colors = colors.copy()
    
    # Greedy polish on best
    best_colors = greedy_polish(best_colors.copy())
    best_count = count_mono_vec(best_colors)
    
    # Simulated annealing with restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Start from best or random
        if restart % 3 == 0:
            colors = best_colors.copy()
        elif restart % 3 == 1:
            colors = random_coloring()
        else:
            colors = thue_morse_coloring()
            # Perturb
            for _ in range(20):
                colors[np.random.randint(n)] ^= 1
        
        cur_count = count_mono_vec(colors)
        if cur_count < best_count:
            best_count = cur_count
            best_colors = colors.copy()
        
        # SA
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        step = 0
        max_steps = 5000
        
        while step < max_steps and T > T_min:
            if time.time() > deadline - 5:
                break
            
            i = np.random.randint(n)
            d = get_delta(colors, i)
            
            if d < 0:
                colors[i] ^= 1
                cur_count += d
                if cur_count < best_count:
                    best_count = cur_count
                    best_colors = colors.copy()
            else:
                # Accept with probability exp(-d/T)
                if d == 0:
                    # Accept neutral moves sometimes
                    if np.random.random() < 0.5:
                        colors[i] ^= 1
                        cur_count += d
                elif np.random.random() < np.exp(-d / T):
                    colors[i] ^= 1
                    cur_count += d
            
            step += 1
            T *= cooling
        
        # Greedy polish after SA
        colors = greedy_polish(colors.copy())
        cur_count = count_mono_vec(colors)
        if cur_count < best_count:
            best_count = cur_count
            best_colors = colors.copy()
    
    # Final greedy polish on best
    best_colors = greedy_polish(best_colors.copy())
    best_count = count_mono_vec(best_colors)
    
    # Return as list of 0/1
    return [int(c) for c in best_colors]
