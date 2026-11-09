"""Stochastic greedy covering for C(19,7,4) using bitmask acceleration."""
import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Stochastic greedy set cover using bitmasks for fast gain calculation.
    """
    start_time = time.time()
    time_limit = 110  # seconds

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    # Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_sub_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Initial uncovered mask: all bits set
    full_mask = (1 << n_t_subsets) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    def _run_trial(seed, alpha=1.0):
        """
        Run a single stochastic greedy trial.
        alpha controls the "temperature" of the probability distribution.
        alpha=1: P(g) proportional to gain.
        alpha=2: P(g) proportional to gain^2 (more greedy).
        alpha=0.5: P(g) proportional to sqrt(gain) (more random).
        """
        rng = random.Random(seed)
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            if time.time() - start_time > time_limit - 1:
                return None, False
                
            # Calculate gains for all blocks
            candidates = []
            weights = []
            
            bm = block_masks
            unc = uncovered
            
            for i in range(n_blocks):
                g = (unc & bm[i]).bit_count()
                if g > 0:
                    candidates.append(i)
                    weights.append(g ** alpha)
            
            if not candidates:
                break
            
            # Select block with probability proportional to weight
            chosen_idx = rng.choices(candidates, weights=weights, k=1)[0]
            blocks.append(all_k_blocks[chosen_idx])
            
            # Update uncovered
            uncovered &= ~block_masks[chosen_idx]
            
        return blocks, True

    # Run multiple trials with different parameters
    alphas = [1.0, 1.5, 2.0, 0.5, 3.0]
    
    for trial in range(500):
        if time.time() - start_time > time_limit - 2:
            break
            
        alpha = alphas[trial % len(alphas)]
        seed = trial * 100 + 7
        
        blocks, success = _run_trial(seed, alpha)
        
        if success and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Fallback if no trial completed
    if best_blocks is None:
        uncovered = full_mask
        blocks = []
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(n_blocks):
                g = (uncovered & block_masks[i]).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_idx = i
            if best_gain <= 0:
                break
            blocks.append(all_k_blocks[best_idx])
            uncovered &= ~block_masks[best_idx]
        best_blocks = blocks
        
    return best_blocks
