"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d with 1 <= a, a+3d <= n
    # In 0-indexed: a0, a0+d, a0+2d, a0+3d where 0 <= a0, a0+3d < n
    # d >= 1
    quads = []
    for d in range(1, n):
        max_a0 = n - 3 * d
        for a0 in range(max_a0):
            quads.append((a0, a0 + d, a0 + 2 * d, a0 + 3 * d))
    quads = np.array(quads, dtype=np.int32)
    num_quads = len(quads)
    
    # Build flat representation for fast bincount
    flat_quads = quads.ravel()  # shape (num_quads * 4,)
    
    def count_mono_vec(color):
        quad_colors = color[quads]  # shape (num_quads, 4)
        min_c = quad_colors.min(axis=1)
        max_c = quad_colors.max(axis=1)
        mono_mask = (min_c == max_c)
        return np.sum(mono_mask)
    
    def get_marginal(color):
        """Compute marginal gain for flipping each bit, vectorized."""
        quad_colors = color[quads]  # (num_quads, 4)
        count_0 = (quad_colors == 0).sum(axis=1)  # (num_quads,)
        
        # change[qi, j] = -1 if mono, +1 if flipping makes mono, 0 otherwise
        change = np.zeros((num_quads, 4), dtype=np.int32)
        
        is_mono = (count_0 == 0) | (count_0 == 4)
        if np.any(is_mono):
            change[is_mono] = -1
        
        mask_c1 = (count_0 == 1)
        if np.any(mask_c1):
            # For quads with exactly one 0, flipping that 0 makes it mono
            change[mask_c1] = (quad_colors[mask_c1] == 0).astype(np.int32)
        
        mask_c3 = (count_0 == 3)
        if np.any(mask_c3):
            # For quads with exactly three 0s (one 1), flipping that 1 makes it mono
            change[mask_c3] = (quad_colors[mask_c3] == 1).astype(np.int32)
        
        # Accumulate marginal for each position
        marginal = np.bincount(flat_quads, weights=change.ravel(), minlength=n).astype(np.int32)
        return marginal
    
    best_score = np.inf
    best_color = None
    
    # Try multiple random restarts
    for seed in range(30):
        rng = np.random.default_rng(seed)
        color = rng.integers(0, 2, size=n).astype(np.int8)
        
        current_score = count_mono_vec(color)
        
        for step in range(2000):
            marginal = get_marginal(color)
            
            flip_mask = marginal < 0
            if not np.any(flip_mask):
                # Local optimum - try random perturbations to escape
                if seed > 0:
                    num_flips = rng.integers(3, 10)
                    flip_indices = rng.integers(0, n, size=num_flips)
                    color[flip_indices] = 1 - color[flip_indices]
                    current_score = count_mono_vec(color)
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
                    continue
                else:
                    break
            
            color[flip_mask] = 1 - color[flip_mask]
            current_score = count_mono_vec(color)
            
            if current_score < best_score:
                best_score = current_score
                best_color = color.copy()
        
        # Early stop if we've found a very good solution
        if best_score <= 10:
            break
    
    return [int(c) for c in best_color]
# EVOLVE-BLOCK-END
