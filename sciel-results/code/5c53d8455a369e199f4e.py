import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices for fast iteration in local search
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights_inv_freq = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)

# Precompute coverage matrix as numpy array for fast vectorized greedy
# coverage_matrix[i, j] = 1 if block i covers subset j
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin below 120s kill

    def fast_greedy(weight_vector):
        """Vectorized weighted greedy using numpy for speed."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        
        while remaining > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            # Multiply coverage matrix by weighted uncovered mask
            gains = coverage_matrix @ (weight_vector * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
            
            chosen.append(best_idx)
            
            # Update uncovered: subsets covered by best_idx are no longer uncovered
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
            
            # Time check inside the loop for safety, though this is fast
            if time.time() > deadline - 5:
                break
        
        return chosen

    def local_search(initial_blocks, seed):
        """Local search: remove redundant blocks, then try swaps."""
        rng = random.Random(seed)
        current_blocks = initial_blocks[:]
        
        # Build counts
        counts = [0] * N_SUBSETS
        for idx in current_blocks:
            for bit in block_bit_indices[idx]:
                counts[bit] += 1
        
        best_len = len(current_blocks)
        best_blocks = current_blocks[:]
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 2:
            # Phase 1: Prune redundant blocks
            pruned = False
            for i in range(len(current_blocks)):
                if time.time() > deadline - 2:
                    break
                idx = current_blocks[i]
                redundant = True
                for bit in block_bit_indices[idx]:
                    if counts[bit] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for bit in block_bit_indices[idx]:
                        counts[bit] -= 1
                    current_blocks.pop(i)
                    pruned = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks[:]
                    no_improve_count = 0
                    break # Restart scan after removal
            
            if pruned:
                continue
                
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perturbation: Add a random block and try to prune
                # Pick a block that covers many currently unique subsets? 
                # Or just random. Let's try random for simplicity and diversity.
                c = rng.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    current_blocks.append(c)
                    for bit in block_bit_indices[c]:
                        counts[bit] += 1
                no_improve_count = 0
                continue
            
            # Phase 2: Try 1-for-1 swap
            # Find a block with few unique subsets to remove
            i = -1
            min_unique = float('inf')
            # Sample some blocks to find a good removal candidate
            sample_size = min(20, len(current_blocks))
            for _ in range(sample_size):
                idx_check = rng.randint(0, len(current_blocks) - 1)
                idx = current_blocks[idx_check]
                unique_count = 0
                for bit in block_bit_indices[idx]:
                    if counts[bit] == 1:
                        unique_count += 1
                if 0 < unique_count < min_unique:
                    min_unique = unique_count
                    i = idx_check
            
            if i == -1:
                # No block with unique subsets found in sample, or all have 0 (redundant, handled above)
                # If all have 0, they are all redundant? No, if min_unique is inf, we didn't find any 0 < u < inf.
                # If a block has 0 unique subsets, it's redundant. Phase 1 should have caught it.
                # So if i is still -1, maybe all blocks have unique subsets >= min_unique?
                # Or maybe we just didn't sample any good ones.
                # Let's just pick a random block if we couldn't find a specific candidate
                i = rng.randint(0, len(current_blocks) - 1)
                idx = current_blocks[i]
                unique_count = 0
                for bit in block_bit_indices[idx]:
                    if counts[bit] == 1:
                        unique_count += 1
                if unique_count == 0:
                    # It is redundant, remove it
                    for bit in block_bit_indices[idx]:
                        counts[bit] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks[:]
                    no_improve_count = 0
                    continue
                elif unique_count > 5:
                    # Too hard to replace, skip
                    continue
                else:
                    # Proceed to find replacement
                    pass
            
            idx = current_blocks[i]
            uncovered_bits = [bit for bit in block_bit_indices[idx] if counts[bit] == 1]
            
            if not uncovered_bits:
                # Should have been caught in Phase 1, but just in case
                for bit in block_bit_indices[idx]:
                    counts[bit] -= 1
                current_blocks.pop(i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks[:]
                no_improve_count = 0
                continue
            
            # Find a replacement block that covers all uncovered_bits
            # Sort uncovered bits by frequency of blocks containing them to speed up intersection
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            
            candidates = list(blocks_containing_subset[uncovered_bits[0]])
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
            
            if candidates:
                # Exclude the block we are removing
                valid_candidates = [c for c in candidates if c != idx]
                if valid_candidates:
                    # Choose candidate that maximizes additional coverage (redundancy reduction potential)
                    best_j = valid_candidates[0]
                    best_additional = -1
                    
                    # Limit search for speed
                    search_limit = min(30, len(valid_candidates))
                    for j in valid_candidates[:search_limit]:
                        # Additional coverage = number of subsets covered by j that are NOT covered by idx
                        # But more importantly, we want to minimize the number of NEW unique subsets it creates?
                        # Or maximize the number of subsets it covers that are currently covered by others (redundant)?
                        # Let's just pick one that covers the most total subsets (greedy heuristic)
                        # Or better: pick one that covers the most subsets with count > 1 (redundant ones)
                        # Actually, just picking a random valid candidate often works well in local search.
                        # Let's try to maximize "gain" in terms of subsets it covers that are currently redundant (count > 1)
                        # This helps in future pruning.
                        
                        # Simpler heuristic: Random selection from valid candidates
                        # This avoids bias towards specific structures
                        best_j = rng.choice(valid_candidates)
                        break # Just pick one
                    
                    # Perform swap
                    # Remove i
                    for bit in block_bit_indices[idx]:
                        counts[bit] -= 1
                    current_blocks.pop(i)
                    
                    # Add best_j
                    current_blocks.append(best_j)
                    for bit in block_bit_indices[best_j]:
                        counts[bit] += 1
                    
                    no_improve_count = 0
                    continue
            
            # If no replacement found, we might get stuck. 
            # The perturbation step (adding random block) will help escape.
            
        return best_blocks, best_len

    # Run multiple greedy initializations with different weightings
    best_overall_blocks = None
    best_overall_len = float('inf')
    
    weight_vectors = [weights_inv_freq, weights_uniform, weights_sqrt_inv]
    
    for w_vec in weight_vectors:
        if time.time() > deadline - 30:
            break
            
        init_blocks = fast_greedy(w_vec)
        
        if not init_blocks:
            continue
            
        # Local search on this initialization
        ls_blocks, ls_len = local_search(init_blocks, seed=42)
        
        if ls_len < best_overall_len:
            best_overall_len = ls_len
            best_overall_blocks = ls_blocks[:]
            
        # If time allows, run local search again with a different seed on the same init?
        # Or just move to next init.
        # Given the speed of vectorized greedy, we can afford more local search iterations.
        # Let's do a second local search pass with a different perturbation seed if time permits.
        if time.time() < deadline - 15:
            ls_blocks2, ls_len2 = local_search(init_blocks, seed=123)
            if ls_len2 < best_overall_len:
                best_overall_len = ls_len2
                best_overall_blocks = ls_blocks2[:]

    if best_overall_blocks is None:
        # Fallback
        init_blocks = fast_greedy(weights_inv_freq)
        best_overall_blocks = init_blocks

    # Final cleanup: remove any remaining redundant blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_overall_blocks:
        for bit in block_bit_indices[idx]:
            final_counts[bit] += 1
            
    final_blocks = best_overall_blocks[:]
    for i in range(len(final_blocks) - 1, -1, -1):
        idx = final_blocks[i]
        redundant = True
        for bit in block_bit_indices[idx]:
            if final_counts[bit] == 1:
                redundant = False
                break
        if redundant:
            for bit in block_bit_indices[idx]:
                final_counts[bit] -= 1
            final_blocks.pop(i)
            
    return [list(all_blocks[i]) for i in final_blocks]
