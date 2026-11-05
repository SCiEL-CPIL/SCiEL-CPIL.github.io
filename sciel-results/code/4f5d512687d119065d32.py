import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subsets it covers
    # block_tsubs[i] is a list of indices of t-subsets covered by block i
    block_tsubs = [[] for _ in range(num_blocks)]
    for i, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            idx = tsub_index[ts]
            block_tsubs[i].append(idx)
            
    # Precompute for each t-subset, the list of blocks that cover it
    # tsub_blocks[j] is a list of indices of blocks that cover t-subset j
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_blocks):
        for j in block_tsubs[i]:
            tsub_blocks[j].append(i)
            
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # GRASP Construction: Probabilistic Greedy
    def grasp_construct(alpha=0.5, rng=None):
        """
        Construct a solution using GRASP.
        At each step, select a block from the restricted candidate list (RCL).
        RCL contains blocks with high 'potential'.
        Potential = gain / (1 + overlap_penalty)
        """
        if rng is None:
            rng = random.Random()
            
        uncovered = set(range(num_tsubs))
        selected_blocks = []
        remaining_blocks = set(range(num_blocks))
        
        # To speed up, we can pre-compute some stats, but for now, let's iterate.
        # We need to find the best gain for each remaining block.
        # gain(b) = |uncovered & block_tsubs[b]|
        
        while uncovered:
            if check_time(10.0):
                break
                
            # Calculate gains for all remaining blocks
            # This is O(N * M) if done naively.
            # Optimization: We can update gains incrementally?
            # When we add a block, we remove its covered t-subsets from 'uncovered'.
            # The gain of other blocks decreases by the number of newly covered t-subsets they shared.
            
            # For simplicity and robustness in Python, let's just recompute gains for candidates.
            # But 75k blocks * 300 steps = 22M ops. 
            # Let's try to be smarter.
            
            # Find max gain
            max_gain = 0
            gains = [0] * num_blocks # Reusing array? No, let's just compute on the fly for RCL.
            
            # Actually, let's just compute the gain for all remaining blocks to find the best and build RCL.
            # This might be slow. Let's sample if it's too slow? 
            # No, let's try the full scan first. If it's too slow, we'll optimize.
            
            current_gains = {}
            for b in remaining_blocks:
                # Count how many of b's t-subsets are in uncovered
                g = 0
                for ts_idx in block_tsubs[b]:
                    if ts_idx in uncovered:
                        g += 1
                current_gains[b] = g
                if g > max_gain:
                    max_gain = g
                    
            if max_gain == 0:
                break # Stuck, shouldn't happen if cover exists
                
            # Build RCL
            # Potential = gain / (1 + overlap)
            # Overlap is tricky to compute on the fly. 
            # Let's simplify: RCL contains blocks with gain >= alpha * max_gain.
            # This is a standard GRASP heuristic.
            
            rcl = []
            for b, g in current_gains.items():
                if g >= alpha * max_gain:
                    rcl.append(b)
                    
            if not rcl:
                rcl = [max(current_gains, key=current_gains.get)]
                
            # Select from RCL probabilistically
            # Uniform random from RCL? Or weighted by gain?
            # Let's use uniform random for diversity.
            chosen = rng.choice(rcl)
            
            selected_blocks.append(chosen)
            remaining_blocks.remove(chosen)
            
            # Update uncovered
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.remove(ts_idx)
                    
        return selected_blocks

    # Local Search: 1-Swap
    def local_search_1swap(initial_blocks, rng=None):
        """
        Try to improve the solution by swapping one block for another.
        A swap (remove b1, add b2) is valid if coverage is maintained.
        We accept if it reduces the number of blocks (impossible in 1-swap, size stays same)
        or if it improves some secondary metric? 
        Actually, 1-swap keeps size constant. 
        To reduce size, we need 1-removal or 2-removal/1-addition.
        
        Let's implement a "Repair-based" local search similar to before but with counts.
        Actually, let's stick to the prompt: "refines them via a 1-swap local search".
        If 1-swap doesn't reduce size, it's useless for minimization unless we combine it with removal.
        
        Let's do a combined move:
        1. Try to remove a block. If coverage breaks, try to add ONE block to fix it.
        If the net change is -1 (removed 1, added 0) -> Improvement.
        If the net change is 0 (removed 1, added 1) -> Neutral (for diversity).
        """
        if rng is None:
            rng = random.Random()
            
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks
            
        # Maintain coverage counts for each t-subset
        # count[j] = number of blocks covering t-subset j
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b in current_blocks:
            for ts_idx in block_tsubs[b]:
                counts[ts_idx] += 1
                
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        T = 10.0
        alpha = 0.99
        iters = 0
        max_iters = 10000 # Keep it small to avoid timeout
        
        while T > 0.1 and iters < max_iters and not check_time(10.0):
            iters += 1
            
            if len(current_blocks) < 2:
                break
                
            # Pick a random block to remove
            remove_idx = rng.randint(0, len(current_blocks) - 1)
            removed_block = current_blocks[remove_idx]
            
            # Tentatively remove
            # Decrease counts
            for ts_idx in block_tsubs[removed_block]:
                counts[ts_idx] -= 1
                
            # Check if coverage is broken
            missing = []
            for j in range(num_tsubs):
                if counts[j] == 0:
                    missing.append(j)
                    
            if not missing:
                # Success! We removed a block and coverage is still full.
                current_blocks.pop(remove_idx)
                best_blocks = list(current_blocks)
                best_len = len(current_blocks)
                T *= alpha
                continue
                
            # Coverage is broken. Try to repair by adding ONE block.
            # We need to find a block that covers ALL missing t-subsets?
            # No, we need to cover the missing ones. If we add a block, it must cover all currently missing t-subsets.
            # If there are multiple missing t-subsets, the new block must contain all of them.
            # This is a strong constraint.
            
            if len(missing) > 1:
                # Check if any single block covers all missing t-subsets
                # This is expensive. 
                # Optimization: If missing is large, it's unlikely.
                # Let's just revert.
                for ts_idx in block_tsubs[removed_block]:
                    counts[ts_idx] += 1
                T *= alpha
                continue
                
            # Only 1 missing t-subset
            missing_ts = missing[0]
            # Find a block that covers missing_ts
            candidates = tsub_blocks[missing_ts]
            # Exclude the removed block? It's not in current_blocks anyway.
            # Pick a random candidate? Or the one with least overlap?
            # Let's pick randomly to maintain diversity.
            if candidates:
                add_block = rng.choice(candidates)
                # Add it
                for ts_idx in block_tsubs[add_block]:
                    counts[ts_idx] += 1
                    
                # We removed 1, added 1. Size is same.
                # Accept with some probability to escape local minima?
                # Or just accept if it's "better" in some sense?
                # Since size is same, let's accept with prob 0.1 to explore.
                if rng.random() < 0.1:
                    # Replace removed_block with add_block
                    current_blocks[remove_idx] = add_block
                    # Counts are already updated (decreased for removed, increased for added)
                    # Wait, we decreased counts for removed_block, then increased for add_block.
                    # This is correct.
                else:
                    # Revert
                    for ts_idx in block_tsubs[add_block]:
                        counts[ts_idx] -= 1
                    for ts_idx in block_tsubs[removed_block]:
                        counts[ts_idx] += 1
            else:
                # No candidate found (shouldn't happen)
                for ts_idx in block_tsubs[removed_block]:
                    counts[ts_idx] += 1
                
            T *= alpha
            
        return best_blocks

    # Main Execution
    best_blocks = []
    best_len = float('inf')
    
    # Run GRASP multiple times with different alphas and seeds
    alphas = [0.3, 0.5, 0.7]
    for seed in range(20):
        if check_time(15.0):
            break
            
        rng = random.Random(seed)
        for alpha in alphas:
            if check_time(5.0):
                break
                
            # Construct
            initial = grasp_construct(alpha, rng)
            
            if not initial:
                continue
                
            # Refine
            refined = local_search_1swap(initial, rng)
            
            if len(refined) < best_len:
                best_len = len(refined)
                best_blocks = refined
                
    # Final check: Ensure validity
    if best_blocks:
        # Verify coverage
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b in best_blocks:
            for ts_idx in block_tsubs[b]:
                counts[ts_idx] += 1
        if np.any(counts == 0):
            # Invalid, fallback to greedy
            # Just return what we have, hoping it's valid? No, must be valid.
            # Let's do a quick greedy repair
            uncovered = set(j for j in range(num_tsubs) if counts[j] == 0)
            while uncovered and not check_time(5.0):
                # Find best block
                best_gain = -1
                best_b = -1
                for b in range(num_blocks):
                    g = 0
                    for ts_idx in block_tsubs[b]:
                        if ts_idx in uncovered:
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_b = b
                if best_b == -1:
                    break
                best_blocks.append(best_b)
                for ts_idx in block_tsubs[best_b]:
                    if ts_idx in uncovered:
                        uncovered.remove(ts_idx)

    return [list(all_blocks[i]) for i in best_blocks]

# To ensure the function is available at module level as required
# The task says "Keep the function names and signatures of the seed program."
# The seed program defines construct(v, k, t).
# I have defined construct above.
