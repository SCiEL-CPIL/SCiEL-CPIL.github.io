"""Randomized greedy set cover with restarts for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct(v, k, t):
    # All candidate k-subsets of {0..v-1}
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Create incidence matrix M: M[b, s] = 1 if block b covers t-subset s
    # Using float32 for faster dot products
    M = np.zeros((n_blocks, n_tsubs), dtype=np.float32)
    
    for b, blk in enumerate(all_blocks):
        for ts in itertools.combinations(blk, t):
            s_idx = tsub_to_idx[ts]
            M[b, s_idx] = 1.0

    def run_greedy(seed):
        rng = np.random.RandomState(seed)
        uncovered = np.ones(n_tsubs, dtype=np.float32)
        used = np.zeros(n_blocks, dtype=bool)
        blocks = []
        
        while uncovered.sum() > 0:
            # Score each unused block: number of uncovered t-subsets it covers
            scores = M.dot(uncovered)
            
            # Mask used blocks
            scores[used] = -1.0
            
            # Add random noise to break ties and allow exploration
            # Randomize the noise scale slightly for diversity
            noise_scale = 0.01 + 0.1 * rng.rand()
            noise = rng.rand(n_blocks).astype(np.float32) * noise_scale
            scores += noise
            
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            
            if best_score <= 0:
                break

            used[best_idx] = True
            blocks.append(list(all_blocks[best_idx]))
            
            # Update uncovered: set to 0 where this block covers
            covered_mask = M[best_idx] > 0.5
            uncovered[covered_mask] = 0.0

        return blocks

    best_blocks = None
    start_time = time.time()
    trial = 0
    
    # Run multiple trials until time limit
    while time.time() - start_time < 100:
        blocks = run_greedy(trial)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            
        trial += 1

    return best_blocks
# EVOLVE-BLOCK-END
