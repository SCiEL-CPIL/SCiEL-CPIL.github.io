"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    quads_list = []
    for d in range(1, n):
        max_a0 = n - 3 * d
        for a0 in range(max_a0):
            quads_list.append((a0, a0 + d, a0 + 2 * d, a0 + 3 * d))
    quads = np.array(quads_list, dtype=np.int32)
    num_quads = len(quads)
    
    def count_mono_vec(color):
        """Count monochromatic 4-term APs using vectorized operations."""
        quad_colors = color[quads]
        min_c = quad_colors.min(axis=1)
        max_c = quad_colors.max(axis=1)
        mono_mask = (min_c == max_c)
        return np.sum(mono_mask)

    def compute_marginals(color):
        """Compute marginal gain for flipping each bit.
        Positive gain means flipping increases the count (bad).
        Negative gain means flipping decreases the count (good).
        """
        quad_colors = color[quads]
        count_0 = (quad_colors == 0).sum(axis=1)
        
        # Initialize change matrix
        # change[qi, j] is the change in objective if we flip quads[qi, j]
        # assuming other bits in the quad are fixed.
        change = np.zeros((num_quads, 4), dtype=np.int8)
        
        # Case 1: Mono quads (count_0 == 0 or 4)
        # Flipping any bit makes it non-mono. Gain -1.
        is_mono = (count_0 == 0) | (count_0 == 4)
        change[is_mono, :] = -1
        
        # Case 2: Non-mono quads
        # count_0 == 1: Three 1s, one 0.
        # Flipping the 0 -> 1111 (Mono). Gain +1.
        # Flipping a 1 -> 0111 (Non-mono). Gain 0.
        mask_c1 = (count_0 == 1)
        if np.any(mask_c1):
            # Positions with color 0 get gain +1
            c1_flip_mask = mask_c1[:, None] & (quad_colors == 0)
            change[c1_flip_mask] = 1
            
        # count_0 == 3: Three 0s, one 1.
        # Flipping the 1 -> 0000 (Mono). Gain +1.
        # Flipping a 0 -> 1000 (Non-mono). Gain 0.
        mask_c3 = (count_0 == 3)
        if np.any(mask_c3):
            # Positions with color 1 get gain +1
            c3_flip_mask = mask_c3[:, None] & (quad_colors == 1)
            change[c3_flip_mask] = 1
            
        # count_0 == 2: Two 0s, two 1s.
        # Flipping any bit keeps it non-mono. Gain 0.
        
        # Sum changes for each position
        marginal = np.zeros(n, dtype=np.int32)
        np.add.at(marginal, quads.flatten(), change.flatten())
        
        return marginal

    def local_search(color, max_iters=200):
        """Perform robust batched local search."""
        best_color = color.copy()
        best_score = count_mono_vec(color)
        current_score = best_score
        stuck_count = 0
        
        for _ in range(max_iters):
            marginal = compute_marginals(color)
            # Find bits with negative marginal gain (flipping them individually reduces objective)
            neg_mask = marginal < 0
            
            if not np.any(neg_mask):
                break
                
            # Strategy: Try flipping subsets of negative gain bits
            # Start with all, then random subsets, then single bits
            candidates = np.where(neg_mask)[0]
            found_improvement = False
            
            # Try 1: Flip all negative gain bits
            # Note: This can be problematic if multiple bits in same quad are flipped,
            # but let's try it first as it's the most aggressive.
            flip_mask = np.zeros(n, dtype=bool)
            flip_mask[candidates] = True
            color[flip_mask] = 1 - color[flip_mask]
            new_score = count_mono_vec(color)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_color = color.copy()
                stuck_count = 0
                found_improvement = True
            else:
                # Revert
                color[flip_mask] = 1 - color[flip_mask]
                
            if not found_improvement:
                # Try 2: Flip random 50% of negative gain bits
                if len(candidates) > 1:
                    num_flip = max(1, len(candidates) // 2)
                    flip_indices = rng.choice(candidates, size=num_flip, replace=False)
                    flip_mask = np.zeros(n, dtype=bool)
                    flip_mask[flip_indices] = True
                    color[flip_mask] = 1 - color[flip_mask]
                    new_score = count_mono_vec(color)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_color = color.copy()
                        stuck_count = 0
                        found_improvement = True
                    else:
                        color[flip_mask] = 1 - color[flip_mask]
                
            if not found_improvement:
                # Try 3: Flip single best bit (most negative marginal)
                if len(candidates) > 0:
                    best_bit = candidates[np.argmin(marginal[candidates])]
                    color[best_bit] = 1 - color[best_bit]
                    new_score = count_mono_vec(color)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_color = color.copy()
                        stuck_count = 0
                        found_improvement = True
                    else:
                        color[best_bit] = 1 - color[best_bit]
                        
            if not found_improvement:
                stuck_count += 1
                if stuck_count > 10:
                    # Try a random perturbation to escape local minimum
                    if rng.random() < 0.1:
                        flip_bit = rng.integers(0, n)
                        color[flip_bit] = 1 - color[flip_bit]
                        new_score = count_mono_vec(color)
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_color = color.copy()
                            stuck_count = 0
                        else:
                            color[flip_bit] = 1 - color[flip_bit]
                    else:
                        break
                        
        return best_color, best_score

    best_overall_color = None
    best_overall_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # 1. Random starts
    for _ in range(20):
        color = rng.integers(0, 2, size=n).astype(np.int8)
        c, s = local_search(color)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()
            
    # 2. Structured starts: Thue-Morse
    # t_n = popcount(n) % 2
    def thue_morse(n):
        # Efficient way to generate Thue-Morse sequence
        seq = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            seq[i] = seq[i >> 1] ^ (i & 1)
        return seq
        
    color = thue_morse(n)
    c, s = local_search(color)
    if s < best_overall_score:
        best_overall_score = s
        best_overall_color = c.copy()
        
    # 3. Stripes
    for width in range(1, 16):
        pattern = np.zeros(n, dtype=np.int8)
        for i in range(n):
            if (i // width) % 2 == 0:
                pattern[i] = 0
            else:
                pattern[i] = 1
        c, s = local_search(pattern)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()
            
    # 4. Modulo patterns
    for mod in range(2, 16):
        color = (np.arange(n) % mod == 0).astype(np.int8)
        c, s = local_search(color)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()
            
        color = (np.arange(n) % mod < 2).astype(np.int8)
        c, s = local_search(color)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()
            
    # 5. Binary patterns
    for shift in range(4):
        color = ((np.arange(n) >> shift) & 1).astype(np.int8)
        c, s = local_search(color)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_color = c.copy()

    return [int(c) for c in best_overall_color]
# EVOLVE-BLOCK-END
