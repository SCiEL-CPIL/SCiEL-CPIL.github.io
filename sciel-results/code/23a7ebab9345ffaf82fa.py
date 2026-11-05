import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for 120s limit
    
    # Precompute all 4-term APs: a, a+d, a+2d, a+3d with d >= 1, all in [0, n-1]
    # Using 0-indexed positions: positions 0..255 correspond to integers 1..256
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Convert to numpy array for vectorized operations
    ap_array = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # Precompute for each position, which APs contain it
    # bit_aps[i] = list of AP indices where position i appears
    bit_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            bit_aps[pos].append(idx)
    
    # Convert to arrays for faster access
    bit_aps_arr = [np.array(l, dtype=np.int32) if l else np.array([], dtype=np.int32) for l in bit_aps]
    
    def count_mono(coloring):
        """Count monochromatic 4-APs. coloring is a numpy array of 0/1."""
        # For each AP, check if all 4 positions have the same color
        colors_at_aps = coloring[ap_array]  # shape (num_aps, 4)
        # Mono if all 4 are equal: (c0==c1) & (c1==c2) & (c2==c3)
        mono = (colors_at_aps[:, 0] == colors_at_aps[:, 1]) & \
               (colors_at_aps[:, 1] == colors_at_aps[:, 2]) & \
               (colors_at_aps[:, 2] == colors_at_aps[:, 3])
        return int(np.sum(mono))
    
    def delta_mono(coloring, pos):
        """Compute change in mono count if we flip position pos.
        Returns (delta, new_mono_count) where delta = new_count - old_count."""
        old_mono = count_mono(coloring)
        new_coloring = coloring.copy()
        new_coloring[pos] = 1 - new_coloring[pos]
        new_mono = count_mono(new_coloring)
        return new_mono - old_mono, new_mono
    
    # Better: incremental delta calculation
    def compute_delta(coloring, pos):
        """Compute the change in mono AP count if we flip position pos.
        Only need to check APs containing pos."""
        # Get all APs containing this position
        ap_indices = bit_aps_arr[pos]
        if len(ap_indices) == 0:
            return 0
        
        # For each AP containing pos, check if it's currently mono and if flipping pos makes it mono
        current_colors = coloring[ap_array[ap_indices]]  # shape (num_affected, 4)
        
        # Find which position in each AP is our pos
        # For each AP, the 4 positions are ap_array[ap_indices, 0..3]
        # We need to find which column has value == pos
        # Actually, let's just check: currently mono? after flip mono?
        
        # Current mono status
        curr_mono = (current_colors[:, 0] == current_colors[:, 1]) & \
                    (current_colors[:, 1] == current_colors[:, 2]) & \
                    (current_colors[:, 2] == current_colors[:, 3])
        
        # After flip: the color at pos changes
        # For each AP, we need to know which index (0,1,2,3) corresponds to pos
        ap_positions = ap_array[ap_indices]  # shape (num_affected, 4)
        
        # Find which column has the value pos
        # Create a mask: for each row, which column equals pos
        is_pos = (ap_positions == pos)  # shape (num_affected, 4)
        
        # New colors after flip
        new_colors = current_colors.copy()
        # Flip the color at the position where is_pos is True
        for col in range(4):
            mask = is_pos[:, col]
            new_colors[mask, col] = 1 - new_colors[mask, col]
        
        new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                   (new_colors[:, 1] == new_colors[:, 2]) & \
                   (new_colors[:, 2] == new_colors[:, 3])
        
        # Delta = (new mono count among affected) - (old mono count among affected)
        # But we only care about the change, and other APs don't change
        delta = int(np.sum(new_mono)) - int(np.sum(curr_mono))
        return delta
    
    # Thue-Morse initial coloring (known to be good for avoiding APs)
    def thue_morse(n):
        """Thue-Morse sequence: t(n) = parity of number of 1s in binary representation of n."""
        result = []
        for i in range(n):
            # Count bits set in i
            bits = bin(i).count('1')
            result.append(bits % 2)
        return np.array(result, dtype=np.int8)
    
    # Generate several initial colorings
    initializations = []
    
    # Thue-Morse
    tm = thue_morse(n)
    initializations.append(tm.copy())
    
    # Thue-Morse complemented
    initializations.append((1 - tm).astype(np.int8))
    
    # Random initializations
    for _ in range(20):
        initializations.append(np.random.randint(0, 2, n).astype(np.int8))
    
    # Alternating blocks of various sizes
    for block_size in [2, 3, 4, 5, 7, 8]:
        coloring = np.zeros(n, dtype=np.int8)
        for i in range(n):
            coloring[i] = (i // block_size) % 2
        initializations.append(coloring.copy())
    
    # Run local search on each initialization
    best_coloring = None
    best_score = float('inf')
    
    def local_search(coloring, max_iters=50000):
        """Hill climbing with random restarts/kicks."""
        current = coloring.copy()
        current_score = count_mono(current)
        best = current.copy()
        best_score_local = current_score
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random position to try flipping
            pos = random.randint(0, n - 1)
            delta = compute_delta(current, pos)
            
            if delta < 0:
                # Improvement
                current[pos] = 1 - current[pos]
                current_score += delta
                no_improve_count = 0
                if current_score < best_score_local:
                    best_score_local = current_score
                    best = current.copy()
            else:
                no_improve_count += 1
                # If stuck for a while, do a random kick
                if no_improve_count > 500:
                    # Flip a random few positions
                    for _ in range(random.randint(1, 5)):
                        kick_pos = random.randint(0, n - 1)
                        current[kick_pos] = 1 - current[kick_pos]
                    current_score = count_mono(current)
                    no_improve_count = 0
        
        return best, best_score_local
    
    # Run local search on each initialization
    for init in initializations:
        if time.time() > deadline:
            break
        
        result, score = local_search(init)
        if score < best_score:
            best_score = score
            best_coloring = result.copy()
    
    # If we have time, do more iterations with the best found so far
    if best_coloring is not None and time.time() < deadline - 5:
        # Continue refining
        result, score = local_search(best_coloring, max_iters=100000)
        if score < best_score:
            best_score = score
            best_coloring = result.copy()
    
    # Final: try some more random kicks and local search
    if best_coloring is not None and time.time() < deadline - 3:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            kick = best_coloring.copy()
            for _ in range(random.randint(1, 8)):
                pos = random.randint(0, n - 1)
                kick[pos] = 1 - kick[pos]
            result, score = local_search(kick, max_iters=20000)
            if score < best_score:
                best_score = score
                best_coloring = result.copy()
    
    # Return as list of ints
    if best_coloring is None:
        # Fallback
        return [(i // 2) % 2 for i in range(256)]
    
    return [int(c) for c in best_coloring]
