import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    # Precompute coverage matrix and frequency
    t_per_block = int(np.math.comb(k, t))
    coverage_matrix = np.zeros((n_blocks, t_per_block), dtype=np.int32)
    
    for b_idx, block in enumerate(all_blocks):
        subsets_in_block = list(itertools.combinations(block, t))
        for j, subset in enumerate(subsets_in_block):
            u_idx = subset_to_idx[subset]
            coverage_matrix[b_idx, j] = u_idx
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Convert block_to_subsets to numpy arrays for faster iteration
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]

    # Calculate frequency of each subset (how many blocks cover it)
    # This is useful for weighted greedy
    freq = np.zeros(n_subsets, dtype=np.float64)
    for u in range(n_subsets):
        freq[u] = len(subset_to_blocks[u])
    
    # Inverse frequency (weight for rare subsets)
    # Add epsilon to avoid division by zero
    inv_freq = 1.0 / (freq + 1e-9)

    def greedy_cover(seed_val, use_weighting=False):
        """Run greedy set cover with optional randomization and weighting. Returns list of block indices."""
        rng = random.Random(seed_val)
        uncovered = np.ones(n_subsets, dtype=bool)
        
        # We maintain a gain array for each block
        # gain[b] = number of uncovered subsets covered by block b
        gains = np.zeros(n_blocks, dtype=np.int32)
        
        # Initial gains: since all are uncovered, gain[b] = t_per_block
        # Wait, gain[b] is the count of *uncovered* subsets in block b.
        # Initially all are uncovered, so gain[b] = t_per_block for all b.
        # But we can be smarter.
        
        # Actually, let's just compute gains dynamically or maintain them.
        # Maintaining them is O(1) update per step if we do it carefully.
        
        # Initialize gains
        # Every block covers t_per_block subsets initially.
        gains[:] = t_per_block
        
        chosen = []
        used = np.zeros(n_blocks, dtype=bool)

        while True:
            if time.time() > deadline:
                break
            if not uncovered.any():
                break

            # Determine best block
            # Mask used blocks
            effective_gains = np.where(used, -1, gains)
            
            if use_weighting:
                # Weighted gain: sum of inv_freq for uncovered subsets in the block
                # This is expensive to compute every step if done naively.
                # Approximation: We can just use standard gain for speed, 
                # or precompute weighted gains and update them.
                # For now, let's stick to standard gain for speed, 
                # but maybe use weighting in tie-breaking?
                # Or just use standard gain. The inspiration suggested weighted greedy.
                # Let's try to implement weighted gain updates.
                # Actually, updating weighted gains is similar to standard gains.
                # weighted_gain[b] = sum(inv_freq[u] for u in block_b if uncovered[u])
                
                # Let's stick to standard gains for the main loop to ensure speed,
                # but perhaps use a different strategy for tie-breaking.
                pass

            max_gain = effective_gains.max()
            if max_gain <= 0:
                break

            # Randomized tie-breaking
            if seed_val is not None and rng.random() < 0.5:
                # Pick randomly among blocks with gain close to max
                threshold = max_gain * 0.9
                candidates = np.where((gains >= threshold) & (~used))[0]
                if len(candidates) > 0:
                    # If too many, sample
                    if len(candidates) > 20:
                        candidates = rng.sample(list(candidates), 20)
                    b_star = candidates[rng.randint(0, len(candidates)-1)]
                else:
                    b_star = int(np.argmax(effective_gains))
            else:
                b_star = int(np.argmax(effective_gains))

            if used[b_star]:
                # Fallback
                gains[b_star] = -1
                b_star = int(np.argmax(gains))
                if gains[b_star] <= 0:
                    break

            chosen.append(b_star)
            used[b_star] = True

            # Update gains
            for u in block_subsets_arr[b_star]:
                if uncovered[u]:
                    uncovered[u] = False
                    # Decrease gain for all blocks that cover u
                    for b in subset_blocks_arr[u]:
                        gains[b] -= 1

            if not uncovered.any():
                break

        return chosen

    def remove_redundant(block_indices):
        """Remove blocks that are not needed for coverage. Returns reduced list."""
        if len(block_indices) <= 1:
            return list(block_indices)

        # Count how many times each subset is covered
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b in block_indices:
            for u in block_subsets_arr[b]:
                cover_count[u] += 1

        # Try removing each block if all its subsets are still covered
        remaining = list(block_indices)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(remaining)):
                b = remaining[i]
                # Check if removing b leaves all its subsets covered
                removable = True
                for u in block_subsets_arr[b]:
                    if cover_count[u] <= 1:
                        removable = False
                        break
                if removable:
                    for u in block_subsets_arr[b]:
                        cover_count[u] -= 1
                    remaining.pop(i)
                    changed = True
                    break  # Restart scan after removal

        return remaining

    def local_search(block_indices):
        """Try to improve by swapping blocks. Returns improved list."""
        if len(block_indices) <= 1:
            return list(block_indices)

        # Build coverage counts
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b in block_indices:
            for u in block_subsets_arr[b]:
                cover_count[u] += 1

        improved = True
        remaining = list(block_indices)
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(remaining)):
                if time.time() > deadline - 2.0:
                    break
                b_remove = remaining[i]
                # Check if b_remove can be removed
                can_remove = True
                for u in block_subsets_arr[b_remove]:
                    if cover_count[u] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    for u in block_subsets_arr[b_remove]:
                        cover_count[u] -= 1
                    remaining.pop(i)
                    improved = True
                    break
                else:
                    # Try swapping: remove b_remove, add a block that covers the critical subsets
                    critical = [u for u in block_subsets_arr[b_remove] if cover_count[u] == 1]
                    if not critical:
                        continue
                    
                    # Find candidate blocks that cover all critical subsets
                    # Heuristic: pick one critical subset and look at blocks covering it
                    # Then check if they cover all others
                    best_candidate = -1
                    best_gain = 0 # Gain is reduction in total blocks? No, we keep size same.
                    # We want to swap if it helps future removals?
                    # Actually, a swap is only useful if it allows a subsequent removal?
                    # Or if it reduces the "tightness"?
                    # Let's just try to find a swap that doesn't break coverage.
                    # If we swap, size stays same.
                    # Is it worth it? Only if it enables a removal later.
                    # So we should try swaps and then run removal again?
                    # That's expensive.
                    
                    # Alternative: Only swap if the new block covers MORE uncovered subsets?
                    # But everything is covered.
                    
                    # Let's stick to simple removal for now. Swaps are risky and slow.
                    # If we can't remove, we leave it.
                    pass

        return remaining

    # Run multiple greedy restarts with different seeds
    best_blocks = None
    best_count = float('inf')

    # 1. Deterministic greedy
    chosen = greedy_cover(None)
    chosen = remove_redundant(chosen)
    chosen = local_search(chosen)
    best_blocks = chosen
    best_count = len(chosen)

    # 2. Randomized restarts
    # We have ~115s total. Precomputation is fast.
    # Greedy is O(N * C(v,k)) which is manageable.
    # Let's run as many as we can.
    
    num_restarts = 50
    for seed_val in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Alternate between weighted and standard? 
        # Let's just use standard with randomization.
        # Maybe try a "lazy" greedy where we don't update all gains, just check?
        # No, standard update is fine.
        
        chosen = greedy_cover(seed_val)
        
        # Quick check: if this is already worse, skip polishing?
        # But polishing might fix it.
        # Let's polish everything.
        
        chosen = remove_redundant(chosen)
        chosen = local_search(chosen)
        
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen

    return [list(all_blocks[b]) for b in best_blocks]
