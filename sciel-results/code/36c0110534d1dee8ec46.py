"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions (0-indexed)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):  # d from 1 to 85
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
    
    def count_mono(coloring):
        """Count monochromatic 4-APs using vectorized operations."""
        c = coloring[aps_arr]  # shape (total_aps, 4)
        return int(np.all(c == c[:, 0:1], axis=1).sum())
    
    def compute_deltas(coloring):
        """Compute delta for flipping each position. Vectorized."""
        c = coloring[aps_arr]  # (total_aps, 4)
        n0 = (c == 0).sum(axis=1)  # number of 0s in each AP
        n1 = 4 - n0
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # Mono APs (all 4 same): flipping any position gives delta -1
        mono_mask = (n0 == 4) | (n0 == 0)
        if mono_mask.any():
            mono_flat = aps_arr[mono_mask].flatten()
            deltas -= np.bincount(mono_flat, minlength=n).astype(np.int32)
        
        # 3-1 APs (n0=3, n1=1): flipping the odd one (color 1) gives delta +1
        mask_31 = (n0 == 3) & (n1 == 1)
        if mask_31.any():
            for idx in np.where(mask_31)[0]:
                ap = aps_arr[idx]
                colors_in_ap = c[idx]
                odd_pos = ap[colors_in_ap == 1][0]
                deltas[odd_pos] += 1
        
        # 1-3 APs (n0=1, n1=3): flipping the odd one (color 0) gives delta +1
        mask_13 = (n0 == 1) & (n1 == 3)
        if mask_13.any():
            for idx in np.where(mask_13)[0]:
                ap = aps_arr[idx]
                colors_in_ap = c[idx]
                odd_pos = ap[colors_in_ap == 0][0]
                deltas[odd_pos] += 1
        
        return deltas
    
    def greedy_improve(coloring, time_limit):
        """Greedy local search: repeatedly flip positions with most negative delta."""
        current_score = count_mono(coloring)
        
        while time.time() < time_limit:
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                break  # Local optimum reached
            
            # Flip all positions with the most negative delta
            flip_mask = (deltas == min_delta)
            flip_positions = np.where(flip_mask)[0]
            
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_score += min_delta * len(flip_positions)
        
        return coloring, current_score
    
    def simulated_annealing(coloring, time_limit):
        """Simulated annealing with vectorized delta computation."""
        current_score = count_mono(coloring)
        best_coloring = coloring.copy()
        best_score = current_score
        
        t = 1.0
        cooling = 0.9995
        min_t = 0.05
        rng = np.random.default_rng()
        
        while time.time() < time_limit:
            if time.time() > time_limit - 3:
                break
            
            # Pick a random position to flip
            pos = rng.integers(0, n)
            
            # Compute delta for this specific position
            deltas = compute_deltas(coloring)
            delta = deltas[pos]
            
            # Accept if improvement or with probability exp(-delta/t)
            if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / t)):
                coloring[pos] = 1 - coloring[pos]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
            
            # Cool down
            t *= cooling
            if t < min_t:
                t = min_t
        
        return best_coloring, best_score
    
    best_coloring = None
    best_score = float('inf')
    
    # Multiple restarts with diverse initializations
    rng = np.random.default_rng(42)
    
    initializations = [
        # Alternating blocks of 2
        np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8),
        # Random
        rng.integers(0, 2, n, dtype=np.int8),
        # All zeros
        np.zeros(n, dtype=np.int8),
        # All ones
        np.ones(n, dtype=np.int8),
        # Alternating
        np.array([i % 2 for i in range(n)], dtype=np.int8),
    ]
    
    # Add perturbed versions of best found so far
    restart_idx = 0
    while time.time() < deadline - 10:
        if restart_idx < len(initializations):
            coloring = initializations[restart_idx].copy()
        else:
            # Perturb the best coloring found so far
            if best_coloring is not None:
                coloring = best_coloring.copy()
                # Flip 10-20 random positions
                n_flips = rng.integers(10, 21)
                flip_positions = rng.choice(n, size=n_flips, replace=False)
                coloring[flip_positions] = 1 - coloring[flip_positions]
            else:
                coloring = rng.integers(0, 2, n, dtype=np.int8)
        
        # Phase 1: Quick greedy improvement
        coloring, score = greedy_improve(coloring.copy(), time_limit=deadline - 8)
        
        # Phase 2: Simulated annealing from the greedy result
        if time.time() < deadline - 5:
            sa_coloring, sa_score = simulated_annealing(coloring.copy(), time_limit=deadline - 3)
            if sa_score < score:
                coloring = sa_coloring
                score = sa_score
        
        # Phase 3: Final greedy polish
        coloring, score = greedy_improve(coloring.copy(), time_limit=deadline - 1)
        
        if score < best_score:
            best_score = score
            best_coloring = coloring.copy()
        
        restart_idx += 1
    
    # Final polish on the best coloring
    if best_coloring is not None and time.time() < deadline - 1:
        best_coloring, best_score = greedy_improve(best_coloring.copy(), time_limit=deadline - 0.5)
    
    if best_coloring is None:
        return [int(x) for x in np.random.randint(0, 2, n)]
    return [int(x) for x in best_coloring]
# EVOLVE-BLOCK-END
