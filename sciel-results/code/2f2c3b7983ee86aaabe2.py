import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
    
    # Build position-to-AP mapping
    pos_to_aps = [[] for _ in range(n)]
    for idx in range(total_aps):
        for pos in aps_arr[idx]:
            pos_to_aps[pos].append(idx)
    pos_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in pos_to_aps]
    
    def count_mono(coloring):
        c = coloring[aps_arr]  # shape (total_aps, 4)
        return int(np.all(c == c[:, 0:1], axis=1).sum())
    
    def compute_deltas(coloring):
        """Compute delta for flipping each position. Vectorized."""
        c = coloring[aps_arr]  # (total_aps, 4)
        n0 = (c == 0).sum(axis=1)  # (total_aps,)
        n1 = 4 - n0
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # Mono APs: flipping any position gives delta -1
        mono_mask = (n0 == 4) | (n1 == 4)
        if mono_mask.any():
            mono_aps = aps_arr[mono_mask]
            mono_flat = mono_aps.flatten()
            deltas -= np.bincount(mono_flat, minlength=n).astype(np.int32)
        
        # 3-1 APs (n0==3, n1==1): flipping the odd one out (color 1) gives delta +1
        mask_31 = (n0 == 3) & (n1 == 1)
        if mask_31.any():
            ap_indices_31 = np.where(mask_31)[0]
            for idx in ap_indices_31:
                ap = aps_arr[idx]
                colors = c[idx]
                odd_pos = ap[colors == 1][0]
                deltas[odd_pos] += 1
        
        # 1-3 APs (n0==1, n1==3): flipping the odd one out (color 0) gives delta +1
        mask_13 = (n0 == 1) & (n1 == 3)
        if mask_13.any():
            ap_indices_13 = np.where(mask_13)[0]
            for idx in ap_indices_13:
                ap = aps_arr[idx]
                colors = c[idx]
                odd_pos = ap[colors == 0][0]
                deltas[odd_pos] += 1
        
        return deltas
    
    def improve_sequential(coloring, max_sweeps=50, time_limit=None):
        """Sequential local search: flip one position at a time (the best one)."""
        current_score = count_mono(coloring)
        
        for sweep in range(max_sweeps):
            if time_limit is not None and time.time() > time_limit:
                break
            
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                break  # No improvement possible
            
            # Find the position with the most negative delta (best single flip)
            flip_pos = int(np.argmin(deltas))
            
            coloring[flip_pos] = 1 - coloring[flip_pos]
            current_score += min_delta
        
        return coloring, current_score
    
    def improve_batch(coloring, max_sweeps=50, time_limit=None):
        """Batch local search: flip all positions with the most negative delta simultaneously."""
        current_score = count_mono(coloring)
        
        for sweep in range(max_sweeps):
            if time_limit is not None and time.time() > time_limit:
                break
            
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                break
            
            flip_mask = (deltas == min_delta)
            flip_positions = np.where(flip_mask)[0]
            
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_score += min_delta * len(flip_positions)
        
        return coloring, current_score
    
    def init_structural():
        """Initialize with a structured pattern that breaks many APs."""
        coloring = np.zeros(n, dtype=np.int32)
        # Use a Thue-Morse-like or modular pattern
        for i in range(n):
            # Count trailing zeros in binary representation of (i+1)
            x = i + 1
            tz = 0
            while x % 2 == 0:
                tz += 1
                x //= 2
            coloring[i] = tz % 2
        return coloring
    
    best_coloring = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # First, try the structural initialization
    coloring = init_structural()
    coloring, score = improve_sequential(coloring, max_sweeps=100, time_limit=limit - 30.0)
    if score < best_score:
        best_score = score
        best_coloring = coloring.copy()
    
    # Then try random restarts with mixed strategies
    while time.time() < limit - 5.0:
        # Initialize random coloring
        coloring = rng.integers(0, 2, n).astype(np.int32)
        
        # Quick batch improvement to get out of the weeds
        coloring, score = improve_batch(coloring, max_sweeps=20, time_limit=limit - 3.0)
        
        # Then sequential polishing for finer optimization
        coloring, score = improve_sequential(coloring, max_sweeps=80, time_limit=limit - 3.0)
        
        if score < best_score:
            best_score = score
            best_coloring = coloring.copy()
        
        # Check if we have time for more
        if time.time() > limit - 10.0:
            break
    
    # Final polishing: do more sequential sweeps on the best coloring
    if best_coloring is not None and time.time() < limit - 2.0:
        for _ in range(3):
            if time.time() > limit - 2.0:
                break
            coloring = best_coloring.copy()
            coloring, score = improve_sequential(coloring, max_sweeps=150, time_limit=limit - 1.0)
            if score < best_score:
                best_score = score
                best_coloring = coloring.copy()
    
    if best_coloring is None:
        return [int(x) for x in np.random.randint(0, 2, n)]
    return [int(x) for x in best_coloring]
# EVOLVE-BLOCK-END
