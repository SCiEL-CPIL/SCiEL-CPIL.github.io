import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
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
    
    # Thue-Morse initial coloring
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # Vectorized count function
    def count_mono(c):
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
                      
    current_score = int(count_mono(colors))
    best_score = current_score
    best_colors = colors.copy()
    
    # Precompute: for each position i, the APs it belongs to and its role
    # aps_containing[i] = list of (ap_idx, pos)
    aps_containing = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in range(4):
            aps_containing[ap[pos]].append((idx, pos))
            
    # For vectorized delta computation, we precompute for each i:
    # - ap_indices_i: array of AP indices containing i
    # - pos_i: array of positions of i in those APs
    # - other_indices: for each AP, the other 3 indices (flattened)
    # This allows us to compute delta for all i at once.
    
    # Actually, let's use a different vectorization strategy:
    # For each flip candidate i, we need to check all APs containing i.
    # We can precompute for each i:
    #   - the set of APs containing i
    #   - for each such AP, the other 3 positions
    # Then we can vectorize over i.
    
    # Let's build arrays for efficient per-i delta calculation
    # For each i, store:
    #   ap_ids[i]: array of AP indices where i appears
    #   pos_in_ap[i]: array of positions (0-3) of i in those APs
    #   other_pos[i]: array of the other 3 positions in those APs (shape (len, 3))
    
    ap_ids = [None] * n
    pos_in_ap = [None] * n
    other_pos = [None] * n
    
    for i in range(n):
        ap_indices = []
        positions = []
        others = []
        for ap_idx, pos in aps_containing[i]:
            ap_indices.append(ap_idx)
            positions.append(pos)
            # Get the other 3 positions
            ap = aps[ap_idx]
            o = [ap[j] for j in range(4) if j != pos]
            others.append(o)
        ap_ids[i] = np.array(ap_indices, dtype=np.int32)
        pos_in_ap[i] = np.array(positions, dtype=np.int8)
        other_pos[i] = np.array(others, dtype=np.int32) if others else np.empty((0,3), dtype=np.int32)
    
    # Vectorized delta computation for all i simultaneously
    def compute_all_deltas(colors):
        """
        Returns an array of length n where delta[i] = change in score if we flip bit i.
        """
        deltas = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            ap_idx_arr = ap_ids[i]
            if len(ap_idx_arr) == 0:
                continue
                
            # Get the APs
            ap_rows = aps_arr[ap_idx_arr]  # shape (K, 4)
            
            # Current colors of all 4 positions in these APs
            b0 = colors[ap_rows[:, 0]]
            b1 = colors[ap_rows[:, 1]]
            b2 = colors[ap_rows[:, 2]]
            b3 = colors[ap_rows[:, 3]]
            
            # Check if currently mono
            is_mono_old = (b0 == b1) & (b1 == b2) & (b2 == b3)
            
            # For each AP, determine the position of i and check if it becomes mono
            # We need to check for each AP: are the other 3 bits equal to (1 - colors[i])?
            
            ci = colors[i]
            new_bit = 1 - ci
            
            # For each AP in ap_idx_arr, we need to check the 3 other positions
            # Let's do this in a loop over the K APs (K is small, ~10-50)
            # Or we can vectorize over the K APs.
            
            # Let's get the other positions for each AP
            # other_pos[i] has shape (K, 3)
            op = other_pos[i]  # shape (K, 3)
            
            # Get the colors of the other 3 positions
            # We need to gather colors for each of the 3 positions
            o0 = colors[op[:, 0]]
            o1 = colors[op[:, 1]]
            o2 = colors[op[:, 2]]
            
            # Check if all 3 others are equal to new_bit
            is_mono_new = (o0 == new_bit) & (o1 == new_bit) & (o2 == new_bit)
            
            # Delta: 
            # -1 if old mono and new not mono
            # +1 if old not mono and new mono
            # 0 otherwise
            delta = np.where(is_mono_old & ~is_mono_new, -1, 
                       np.where(~is_mono_old & is_mono_new, 1, 0))
            
            deltas[i] = np.sum(delta)
        
        return deltas
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    while time.time() < deadline:
        # Compute deltas for all positions
        deltas = compute_all_deltas(colors)
        
        # Choose a random position to flip
        i = np.random.randint(n)
        d = deltas[i]
        
        # Accept if improvement, or with probability exp(-d/T) if worsening
        if d < 0:
            accept = True
        elif d == 0:
            accept = True
        else:
            accept = (np.random.rand() < np.exp(-d / T))
        
        if accept:
            colors[i] = 1 - colors[i]
            current_score += d
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    return [int(c) for c in best_colors]
