import itertools
import numpy as np
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_set = set(all_t_subs)

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    # Map t-subset to index
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # For each block, compute which t-subsets it covers
    # block_to_tsubs: list of lists of t-subset indices
    block_to_tsubs = []
    for block in all_k_blocks:
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_to_tsubs.append(subs)

    def time_left():
        return deadline - time.time()

    def greedy_set_cover(uncovered_indices, available_blocks, seed=None):
        """
        Greedy set cover.
        uncovered_indices: set of t-subset indices to cover
        available_blocks: list of block indices (or all if None)
        Returns list of block indices.
        """
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(uncovered_indices)
        if not uncovered:
            return []
        
        # If available_blocks is None, use all blocks
        if available_blocks is None:
            block_pool = list(range(n_k_blocks))
        else:
            block_pool = list(available_blocks)
            
        random.shuffle(block_pool)
        
        selected_blocks = []
        
        # For efficiency, we maintain a count of how many uncovered t-subsets each block covers
        # But recomputing gain for all blocks each step is O(n_blocks * C(k,t)) which is heavy.
        # For v=19, k=7, t=4: C(7,4)=35. n_blocks=50388.
        # 50388 * 35 = ~1.7M ops per step.
        # If we need ~100 blocks, that's 170M ops. In Python, this might be too slow (10-20s).
        # We need a faster heuristic or numpy acceleration.
        
        # Let's use a randomized greedy with limited lookahead or just pick from a random subset of blocks?
        # Or use numpy to compute gains?
        
        # Strategy:
        # 1. Start with a random permutation of blocks.
        # 2. At each step, pick the block from the remaining pool that covers the most uncovered t-subsets.
        #    To speed this up, we can sample a random subset of the remaining pool if it's large, 
        #    or use a "lazy" update strategy.
        
        # Given the constraints, let's try a standard greedy but optimized with numpy where possible.
        # However, dynamic updates of "uncovered" make vectorization hard.
        
        # Alternative: Use a "greedy packing" first to reduce the problem size?
        # Or just run multiple short greedy runs with different random seeds and keep the best.
        
        # Let's try a simple randomized greedy:
        # At each step, pick a random block from the pool, but biased towards those covering more uncovered?
        # Actually, pure random greedy is often decent for small instances.
        
        # Let's implement a faster greedy:
        # Maintain a list of (gain, block_idx).
        # Initially compute gain for all blocks.
        # Pick max gain.
        # Update gains for remaining blocks (decrement if they cover the newly covered t-subsets).
        # This is still O(n_blocks * C(k,t)) per step.
        
        # Let's just run a few iterations of a simple random greedy and a standard greedy.
        
        # Standard Greedy with NumPy acceleration for gain calculation?
        # We can precompute a sparse matrix or just use lists.
        
        # Let's try a "Greedy with Randomization":
        # 1. Compute gain for all blocks.
        # 2. Pick top-K blocks randomly? No, pick the best.
        # 3. To avoid O(N^2) total time, we might need to be clever.
        
        # For now, let's implement a basic greedy that is reasonably fast.
        # If it's too slow, we fall back to random selection.
        
        # Optimization: Only consider blocks that cover at least one uncovered t-subset.
        
        while uncovered and time_left() > 5:
            best_block = -1
            best_gain = 0
            
            # Iterate through a random subset of blocks to find a good one?
            # Or iterate all?
            # Let's iterate all, but break early if time is low.
            
            for b_idx in block_pool:
                if b_idx < 0: # removed
                    continue
                # Calculate gain
                gain = 0
                for sub_idx in block_to_tsubs[b_idx]:
                    if sub_idx in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we can't beat current best? No, need to check all.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break # No block covers any uncovered subset? Should not happen if covering exists.
                
            selected_blocks.append(best_block)
            
            # Remove covered t-subsets
            for sub_idx in block_to_tsubs[best_block]:
                uncovered.discard(sub_idx)
            
            # Remove best_block from pool (or mark as used)
            # For simplicity, we don't strictly need to remove it if we allow duplicates?
            # No, blocks must be distinct. So we should not pick the same block twice.
            # We can just remove it from the pool list.
            # But removing from a list is O(N).
            # Let's just keep a "used" set.
            
            # Actually, the loop above iterates block_pool. If we don't remove, we might pick the same block again.
            # So we need to track used blocks.
            
            # Let's refactor: keep a list of available blocks, and remove the selected one.
            # This is inefficient.
            
            # Better: Use a priority queue?
            # Or just accept the O(N) removal cost. N=50k. 100 steps * 50k = 5M ops. Acceptable.
            
            # But the gain calculation is the bottleneck.
            # Let's try to speed up gain calculation.
            
        return selected_blocks

    # The above greedy is likely too slow in pure Python for 50k blocks.
    # Let's use a different strategy:
    # 1. Generate a random set of blocks (e.g., 2000 random blocks).
    # 2. Use greedy set cover on this smaller pool.
    # 3. Repeat with different random pools.
    # 4. Combine results and do local search.
    
    def fast_greedy_on_pool(pool_indices, uncovered_indices):
        """
        Greedy set cover on a specific pool of blocks.
        pool_indices: list of block indices
        uncovered_indices: set of t-subset indices
        """
        uncovered = set(uncovered_indices)
        if not uncovered:
            return []
        
        available = list(pool_indices)
        random.shuffle(available)
        
        selected = []
        
        while uncovered and time_left() > 2:
            best_block = -1
            best_gain = 0
            
            for b_idx in available:
                gain = 0
                for sub_idx in block_to_tsubs[b_idx]:
                    if sub_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            
            for sub_idx in block_to_tsubs[best_block]:
                uncovered.discard(sub_idx)
            
            available.remove(best_block)
            
        return selected

    def local_search(blocks, max_iter=100):
        """
        Try to improve the covering by replacing blocks or removing redundant ones.
        """
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        current_set = set(current_blocks)
        
        # Compute coverage counts
        # t_sub_to_blocks: dict from t_sub_idx to list of block indices covering it
        # Or just a count.
        
        # Let's do a simple removal first.
        # Check if any block is redundant.
        
        # To check redundancy, we need to know if every t-subset in block B is covered by some other block.
        # This is expensive.
        
        # Alternative: Try to replace a block with a different one that covers the same "critical" t-subsets.
        
        # Let's try a simple hill climbing:
        # 1. Try removing each block. If valid, keep removed.
        # 2. Try swapping each block with a random other block. If improves (smaller count? no, same count, but maybe allows future removals?), keep.
        
        # Since we minimize count, removal is the key.
        
        # Efficient removal check:
        # Maintain a count of how many blocks cover each t-subset.
        
        counts = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in current_blocks:
            for sub_idx in block_to_tsubs[b_idx]:
                counts[sub_idx] += 1
        
        changed = True
        while changed and time_left() > 2:
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                # Tentatively remove b_idx
                for sub_idx in block_to_tsubs[b_idx]:
                    counts[sub_idx] -= 1
                
                # Check if any count is 0
                if np.any(counts == 0):
                    # Cannot remove
                    for sub_idx in block_to_tsubs[b_idx]:
                        counts[sub_idx] += 1
                    new_blocks.append(b_idx)
                else:
                    # Can remove
                    changed = True
                    # Do not add to new_blocks
            
            current_blocks = new_blocks
            
        return current_blocks

    def random_greedy_run(seed):
        random.seed(seed)
        
        # Pick a random pool of blocks
        pool_size = min(n_k_blocks, 2000)
        pool = random.sample(range(n_k_blocks), pool_size)
        
        # Add all blocks that cover specific "hard" t-subsets?
        # For now, just random.
        
        uncovered = set(range(n_t_subs))
        
        blocks = fast_greedy_on_pool(pool, uncovered)
        
        # If not covered, add more blocks
        # Check what's left
        covered = set()
        for b_idx in blocks:
            for sub_idx in block_to_tsubs[b_idx]:
                covered.add(sub_idx)
        
        remaining = set(range(n_t_subs)) - covered
        
        if remaining:
            # Greedily add blocks from the full set to cover remaining
            # This might be slow, but remaining should be small if pool was good.
            # Use a simple loop
            while remaining and time_left() > 2:
                best_block = -1
                best_gain = 0
                # Sample a few blocks
                candidates = random.sample(range(n_k_blocks), min(500, n_k_blocks))
                for b_idx in candidates:
                    if b_idx in blocks:
                        continue
                    gain = 0
                    for sub_idx in block_to_tsubs[b_idx]:
                        if sub_idx in remaining:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                
                if best_block == -1:
                    break
                    
                blocks.append(best_block)
                for sub_idx in block_to_tsubs[best_block]:
                    remaining.discard(sub_idx)
                    
        # Local search
        blocks = local_search(blocks)
        
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random greedy instances
    num_runs = 10
    
    for i in range(num_runs):
        if time_left() < 5:
            break
            
        blocks = random_greedy_run(i)
        
        if blocks is None:
            continue
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Final local search on the best
    if best_blocks:
        best_blocks = local_search(best_blocks)
        
    # Convert indices back to actual blocks (lists of points)
    if best_blocks:
        result = [list(all_k_blocks[b_idx]) for b_idx in best_blocks]
        # Ensure distinct
        # The algorithm ensures distinct indices, so distinct blocks.
        return result
    else:
        return []
