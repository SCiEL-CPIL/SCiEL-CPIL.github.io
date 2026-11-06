import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0  # Safety margin before 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with d>=1, a+3d <= n
    # Using 0-based indexing: positions 0..255
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # For each position, store list of (ap_index, other_3_indices)
    # other_3_indices are the other 3 positions in the AP
    pos_aps = [[] for _ in range(n)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append((ap_idx, (i1, i2, i3)))
        pos_aps[i1].append((ap_idx, (i0, i2, i3)))
        pos_aps[i2].append((ap_idx, (i0, i1, i3)))
        pos_aps[i3].append((ap_idx, (i0, i1, i2)))
    
    # Convert to numpy arrays for faster access
    # For each position, store the ap indices and the other 3 indices as a flat array
    pos_ap_indices = [np.array([x[0] for x in pos_aps[i]], dtype=np.int32) for i in range(n)]
    pos_other3 = [np.array([idx for x in pos_aps[i] for idx in x[1]], dtype=np.int32) for i in range(n)]
    
    def count_mono(color):
        """Count monochromatic 4-term APs."""
        count = 0
        for (i0, i1, i2, i3) in aps:
            c = color[i0]
            if c == color[i1] == color[i2] == color[i3]:
                count += 1
        return count
    
    def delta_flip(color, pos):
        """Compute change in monochromatic AP count if we flip position pos."""
        ap_indices = pos_ap_indices[pos]
        others = pos_other3[pos]
        if len(ap_indices) == 0:
            return 0
        
        cur_color = color[pos]
        # For each AP containing pos, check if the other 3 are all same as cur_color or all opposite
        # Reshape others to (-1, 3)
        others_reshaped = others.reshape(-1, 3)
        
        # Check if all 3 others have same color as cur_color
        match_cur = np.all(others_reshaped == cur_color, axis=1)
        # Check if all 3 others have opposite color
        match_opp = np.all(others_reshaped == (1 - cur_color), axis=1)
        
        # If all 3 match cur_color: flipping removes one mono AP -> delta = -1
        # If all 3 match opp: flipping creates one mono AP -> delta = +1
        # Otherwise: no change
        delta = np.sum(match_opp) - np.sum(match_cur)
        return int(delta)
    
    def local_search(color, max_iter=5000):
        """Delta-based local search."""
        color = color.copy()
        current_count = count_mono(color)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 5:
            improved = False
            iters += 1
            # Try all positions, pick the one with best (most negative) delta
            best_pos = -1
            best_delta = 0
            # Shuffle for randomization
            order = list(range(n))
            random.shuffle(order)
            for pos in order:
                d = delta_flip(color, pos)
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
            if best_pos >= 0 and best_delta < 0:
                color[best_pos] = 1 - color[best_pos]
                current_count += best_delta
                improved = True
        return color, current_count
    
    # Start with Thue-Morse sequence
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    # Best solution found
    best_color = thue_morse(n)
    best_count = count_mono(best_color)
    
    # Run local search from Thue-Morse
    color, cnt = local_search(best_color, max_iter=10000)
    if cnt < best_count:
        best_color = color.copy()
        best_count = cnt
    
    # Multiple restarts with random perturbations
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Start from best or random
        if r % 3 == 0:
            start = best_color.copy()
        else:
            start = thue_morse(n).copy()
        
        # Random perturbation: flip some random positions
        num_flips = random.randint(5, 30)
        flip_positions = random.sample(range(n), num_flips)
        for pos in flip_positions:
            start[pos] = 1 - start[pos]
        
        # Local search
        color, cnt = local_search(start, max_iter=3000)
        if cnt < best_count:
            best_color = color.copy()
            best_count = cnt
    
    # Final local search from best
    if time.time() < deadline - 5:
        color, cnt = local_search(best_color, max_iter=20000)
        if cnt < best_count:
            best_color = color.copy()
            best_count = cnt
    
    return [int(c) for c in best_color]
