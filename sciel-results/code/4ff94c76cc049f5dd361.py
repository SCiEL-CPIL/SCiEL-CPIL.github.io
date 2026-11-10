import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 112.0

def _generic_greedy(v, k, t, start_time):
    """Generic randomized greedy for small instances."""
    points = list(range(v))
    all_t = list(itertools.combinations(points, t))
    n_t = len(all_t)
    idx_map = {s: i for i, s in enumerate(all_t)}
    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)

    # coverage[b][s] = 1 if block b covers t-subset s
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    for b, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            cov[b, idx_map[sub]] = 1

    uncovered = np.ones(n_t, dtype=bool)
    chosen = []
    deadline = start_time + DEADLINE_SECONDS

    while uncovered.any() and time.time() < deadline:
        gains = cov[:, uncovered].sum(axis=1)
        if gains.max() <= 0:
            break
        b = int(np.argmax(gains))
        chosen.append(b)
        uncovered[cov[b] == 1] = False

    return [list(all_blocks[i]) for i in chosen]

def construct(v, k, t):
    start_time = time.time()

    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    # Precompute for v=19, k=8, t=5
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)  # 11628
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)  # 75582

    # Build coverage matrix: cov[b, s] = 1 if block b covers t-subset s
    # Each block of size 8 covers C(8,5)=56 t-subsets
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    cov_f32 = np.zeros((n_b, n_t), dtype=np.float32)
    # Also store the subset indices for each block for faster pruning/swapping
    block_subsets = []
    for b, blk in enumerate(all_blocks):
        subs = []
        for sub in itertools.combinations(blk, t):
            idx = idx_map[sub]
            cov[b, idx] = 1
            cov_f32[b, idx] = 1.0
            subs.append(idx)
        block_subsets.append(subs)

    # For each t-subset, list of blocks that cover it
    blocks_for_subset = []
    for s in range(n_t):
        blocks_for_subset.append(np.where(cov[:, s] == 1)[0])

    best_blocks = None
    best_count = float('inf')
    deadline = start_time + DEADLINE_SECONDS

    rng = random.Random(42)

    # Helper function to prune
    def prune(chosen_list):
        if not chosen_list:
            return []
        
        # Count coverage for each t-subset
        counts = np.zeros(n_t, dtype=np.int32)
        for b in chosen_list:
            subs = block_subsets[b]
            if subs:
                counts[subs] += 1
        
        new_chosen = []
        # To make pruning order-independent or better, we can do multiple passes
        # or just one pass in random order. One pass is O(N).
        # A block is removable if for all s in block, counts[s] > 1.
        # We can check this efficiently.
        
        # Shuffle to avoid bias
        indices = list(range(len(chosen_list)))
        rng.shuffle(indices)
        
        for i in indices:
            b = chosen_list[i]
            # Check if b is redundant
            # Subsets covered by b are where cov[b] == 1
            # We need counts[s] > 1 for all s where cov[b][s] == 1
            # This is equivalent to: min(counts[cov[b]==1]) > 1
            # But cov[b]==1 is a sparse mask.
            # Let's use the precomputed block_subsets
            subs = block_subsets[b]
            if subs:
                # Check counts for these subsets
                # If any count is 1, b is essential
                if np.any(counts[subs] == 1):
                    new_chosen.append(b)
                else:
                    # Redundant, remove
                    counts[subs] -= 1
            else:
                # Should not happen for valid blocks
                new_chosen.append(b)
        
        return new_chosen

    while time.time() < deadline - 5.0:
        # Randomized greedy with random tie-breaking
        uncovered = np.ones(n_t, dtype=np.float32)
        chosen = []

        while uncovered.any():
            # Pick a random uncovered t-subset
            unc_idx = np.where(uncovered > 0.5)[0]
            if len(unc_idx) == 0:
                break
            s = int(rng.choice(unc_idx))

            # Candidate blocks covering this subset
            cands = blocks_for_subset[s]
            if len(cands) == 0:
                break

            # Compute gain for each candidate: number of uncovered subsets it covers
            # Use float32 for faster dot product
            gains = np.einsum('ij,j->i', cov_f32[cands], uncovered)

            # Add small random perturbation to break ties
            # Generate random noise for the candidates
            noise = rng.random() * 0.5
            gains_perturbed = gains + noise

            best_local = int(np.argmax(gains_perturbed))
            b = int(cands[best_local])

            chosen.append(b)
            # Update uncovered: set to 0 where cov[b] is 1
            # cov_f32[b] is a float32 array
            uncovered[cov_f32[b] == 1.0] = 0.0

        # Apply pruning
        pruned = prune(chosen)
        
        if len(pruned) < len(chosen):
            chosen = pruned
        
        # Check if we found a better solution
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen[:]

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = np.ones(n_t, dtype=np.float32)
        chosen = []
        while uncovered.any() and time.time() < deadline:
            gains = np.einsum('ij,j->i', cov_f32, uncovered)
            b = int(np.argmax(gains))
            if gains[b] <= 0:
                break
            chosen.append(b)
            uncovered[cov_f32[b] == 1.0] = 0.0
        best_blocks = chosen
        # Prune the fallback
        best_blocks = prune(best_blocks)

    return [list(all_blocks[i]) for i in best_blocks]
