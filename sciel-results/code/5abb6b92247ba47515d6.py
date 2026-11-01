"""Covering design C(v,k,t): Optimized Greedy Set Cover with Count-Based Minimalization and Swap."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # Safety margin

    # --- Precompute t-subset index table ---
    all_tsubs = list(itertools.combinations(range(v), t))
    n_t = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # --- Precompute all k-subsets and their t-subset index lists ---
    all_ksubs = list(itertools.combinations(range(v), k))
    n_k = len(all_ksubs)
    
    # Store t-subset indices for each k-subset
    ksub_tsets = []
    for comb in all_ksubs:
        tset = []
        for c in itertools.combinations(comb, t):
            tset.append(tsub_to_idx[c])
        ksub_tsets.append(tset)
        
    # Also store the k-subsets themselves for output
    # We don't strictly need masks if we use counts, but masks are useful for quick "is this block useful" checks?
    # With counts, we don't need global masks for minimalization.
    
    # --- State ---
    blocks = []
    block_indices = []
    used = np.zeros(n_k, dtype=np.int8)
    
    # Coverage count for each t-subset
    t_cover_count = np.zeros(n_t, dtype=np.int16)
    
    # Uncovered mask for greedy selection (bit i set if t-subset i is uncovered)
    # Since n_t = 3876, we cannot use a single Python int bit mask efficiently for operations like `&` if we do it naively?
    # Actually, Python big ints are quite fast for bitwise ops.
    # Let's use a Python int for the uncovered mask.
    uncovered_mask = (1 << n_t) - 1
    remaining = n_t

    # Weights for Lagrangian-guided greedy
    w = np.ones(n_t, dtype=np.float64)

    # --- Greedy Phase ---
    max_rounds = 300
    for rnd in range(max_rounds):
        if time.time() > deadline - 20.0:
            break
        if remaining == 0:
            break

        # Find best block
        best_idx = -1
        best_gain = 0.0

        # Optimization: Iterate through blocks. 
        # To speed up, we can skip blocks that are "dominated" or simply rely on the loop.
        # The inner loop iterates over ~35 items. 50k * 35 = 1.75M ops per round.
        # 150 rounds * 1.75M = 262M ops. This might be too slow in pure Python (approx 10-20s).
        # We need to optimize the gain calculation.
        
        # Alternative: Pre-compute nothing, just loop.
        # Let's try to vectorize the gain calculation if possible, or just accept the cost if it's under 5s.
        # 262M simple float adds/comparisons in Python is likely ~10-15 seconds. 
        # We have 145s budget. So 15s for greedy is fine.
        
        for i in range(n_k):
            if used[i]:
                continue
            
            gain = 0.0
            # Unroll or use local variables for speed
            for idx in ksub_tsets[i]:
                if (uncovered_mask >> idx) & 1:
                    gain += w[idx]
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1 or best_gain <= 0:
            break

        # Add block
        blocks.append(list(all_ksubs[best_idx]))
        block_indices.append(best_idx)
        used[best_idx] = 1
        
        # Update coverage counts and mask
        for idx in ksub_tsets[best_idx]:
            t_cover_count[idx] += 1
            if t_cover_count[idx] == 1:
                uncovered_mask &= ~(1 << idx)
        
        remaining = uncovered_mask.bit_count()

        # Update weights (Lagrangian relaxation)
        # Simplified update: increase weight of uncovered, decrease covered?
        # Or just keep weights uniform if the complexity is too high?
        # The current code updates weights every round which is O(n_t) = 3876 ops. Negligible.
        step = 2.0 / (rnd + 1.0)
        # Vectorized weight update
        # Create a boolean mask of uncovered
        # This is hard to do without converting int mask to array.
        # Let's just do a simple heuristic: weights remain 1.0? 
        # The prompt says "Lagrangian-guided". Let's keep the weight update but make it faster.
        # Actually, for small v, uniform greedy is often very strong. 
        # Let's stick to the current weight update logic but ensure it's not the bottleneck.
        
        # Efficient weight update:
        # w[uncovered] += step
        # w[covered] -= step * 0.5
        # We can iterate over all_tsubs? No, that's 3876. 
        # Or iterate over ksub_tsets of the new block to update covered? 
        # The previous code iterated over ALL n_t. That's 3876 * 150 = 580k ops. Fine.
        
        for idx in range(n_t):
            if (uncovered_mask >> idx) & 1:
                w[idx] += step
            else:
                w[idx] -= step * 0.5
        w = np.maximum(w, 0.1)
        w /= w.mean()

    # --- Minimalization using Counts ---
    # A block is redundant if for all t in its tset, count[t] > 1
    changed = True
    while changed:
        if time.time() > deadline - 10.0:
            break
        changed = False
        new_blocks = []
        new_block_indices = []
        
        for i, blk_idx in enumerate(block_indices):
            tset = ksub_tsets[blk_idx]
            essential = False
            for idx in tset:
                if t_cover_count[idx] == 1:
                    essential = True
                    break
            
            if not essential:
                # Remove block
                changed = True
                # Decrement counts
                for idx in tset:
                    t_cover_count[idx] -= 1
                used[blk_idx] = 0
            else:
                new_blocks.append(blocks[i])
                new_block_indices.append(blk_idx)
        
        blocks = new_blocks
        block_indices = new_block_indices

    # --- Swap Local Search using Counts ---
    # Try to replace block i with unused block j if it covers the "lost" t-subsets of i
    if time.time() < deadline - 5.0:
        improved = True
        max_swap_iters = 10
        for _ in range(max_swap_iters):
            if time.time() > deadline - 2.0:
                break
            if not improved:
                break
            improved = False
            
            n_blocks = len(blocks)
            for i in range(n_blocks):
                if time.time() > deadline - 2.0:
                    break
                
                blk_idx = block_indices[i]
                tset_i = ksub_tsets[blk_idx]
                
                # Simulate removal of block i
                lost_tsubs = []
                for idx in tset_i:
                    if t_cover_count[idx] == 1:
                        lost_tsubs.append(idx)
                
                if not lost_tsubs:
                    continue # Should have been removed in minimalization
                
                # Find best replacement
                best_swap_idx = -1
                best_swap_gain = 0
                
                # To speed up, we can pre-filter candidates? 
                # Just loop through unused blocks.
                for j in range(n_k):
                    if used[j]:
                        continue
                    
                    tset_j = ksub_tsets[j]
                    gain = 0
                    for idx in tset_j:
                        if idx in lost_tsubs: # This check is O(1) if lost_tsubs is a set, but it's a list.
                            # Better: use a boolean array for lost_tsubs?
                            # lost_tsubs is small (usually 1-5 items). List scan is fine.
                            gain += 1
                    
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_idx = j
                
                if best_swap_idx != -1 and best_swap_gain > 0:
                    # Perform swap
                    # Remove i
                    for idx in tset_i:
                        t_cover_count[idx] -= 1
                    used[blk_idx] = 0
                    
                    # Add j
                    for idx in ksub_tsets[best_swap_idx]:
                        t_cover_count[idx] += 1
                    used[best_swap_idx] = 1
                    
                    blocks[i] = list(all_ksubs[best_swap_idx])
                    block_indices[i] = best_swap_idx
                    improved = True
                    break # Restart the loop over blocks? Or continue? 
                    # Breaking restarts the outer while loop, which is safer for convergence.
            
            # If we broke inner loop, we restart while loop
            # If we didn't break, we finished one pass with no improvements

    return blocks
# EVOLVE-BLOCK-END
