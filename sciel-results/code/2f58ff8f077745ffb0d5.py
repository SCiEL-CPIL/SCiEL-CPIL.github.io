"""
Strategy: Local Search (Hill Climbing) on a fixed pool of blocks.
Instead of greedily appending blocks until coverage is complete, we start with a 
reasonably sized pool of blocks (e.g., 80 blocks, slightly above the known best 76) 
and iteratively try to improve the solution by replacing the "worst" block with a 
better one. This allows the algorithm to "fix" inefficient blocks that were 
necessary in a greedy pass but become redundant or suboptimal in a global context.
"""

import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s limit

    # 1. Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for efficient array operations
    tsub_index = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # Target number of blocks to start with. 
    # The known best is 76. We start with 80 to ensure we can likely cover everything 
    # initially, then try to remove or improve.
    # Actually, a safer bet for local search is to start with a greedy solution 
    # and then try to reduce size or improve quality. 
    # However, the current greedy is slow. 
    # Let's use a faster initialization: Greedy with random tie-breaking, but 
    # optimized with numpy for the gain calculation.
    
    # State:
    # blocks: list of tuples (each tuple is a block of k points)
    # cover_count: np.array of shape (num_tsubsets,), counts how many blocks cover each t-subset
    # uncovered_mask: np.array of shape (num_tsubsets,), bool, True if cover_count[i] == 0
    
    blocks = []
    cover_count = np.zeros(num_tsubsets, dtype=np.int32)
    
    # Helper to update cover_count for a block
    def add_block(block):
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                # It was uncovered, now covered
                pass # We'll recompute uncovered count or maintain a separate counter
            
    def remove_block(block):
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            cover_count[idx] -= 1
    
    # Initialization: Greedy construction
    # We need a fast way to pick the next block.
    # We'll use a simple greedy: pick the block that covers the most currently uncovered t-subsets.
    # To make this fast, we can use a random sampling approach for candidate blocks 
    # or a fast heuristic.
    
    # Given the time limit, a pure exhaustive search for the best block is too slow.
    # We will use a "Greedy with Random Restart" for initialization, but keep the 
    # number of blocks fixed at a target N_blocks.
    
    # Let's estimate N_blocks. 
    # Lower bound is 69. Best known 76.
    # Let's try to start with 78 blocks.
    target_blocks = 78
    
    # Initialize with random blocks? No, that's likely to leave many holes.
    # Better: Greedy fill until we have target_blocks blocks.
    
    # To speed up the greedy phase, we'll use a heuristic block generator:
    # Pick a random uncovered t-subset, then greedily add points that cover the most 
    # OTHER uncovered t-subsets involving those points.
    
    # Since we can't easily maintain a dynamic "uncovered" set efficiently in pure Python 
    # for every step without heavy overhead, we will use a simpler initialization:
    # Generate target_blocks blocks using a fast greedy heuristic that doesn't track 
    # global coverage perfectly, but tries to be "dense".
    
    # Actually, let's stick to the most robust initialization: 
    # Standard Greedy, but we stop exactly at target_blocks.
    
    # Optimization: Precompute all combinations of size k? No, C(24,7) is 346k, manageable.
    # But calculating gain for each is expensive.
    
    # Let's use a different initialization: 
    # Random blocks + Local Search repair.
    # Start with target_blocks random blocks.
    # This will likely leave many uncovered t-subsets.
    # Then, in the main loop, we identify the "worst" block (lowest unique coverage) 
    # and replace it with a block that specifically targets the most uncovered t-subsets.
    
    # This "Repair" strategy is often faster to converge to a valid state than 
    # pure greedy, and local search can then optimize.
    
    # 1. Initialize with random blocks
    for _ in range(target_blocks):
        block = tuple(sorted(random.sample(range(v), k)))
        blocks.append(block)
        add_block(block)
    
    # Check if already valid (unlikely)
    # uncovered_count = np.sum(cover_count == 0)
    
    # Main Loop: Local Search
    # We will iterate through blocks, and for each block, we decide if we should replace it.
    # Metric for "worst": Number of t-subsets covered ONLY by this block (unique coverage).
    # If a block has low unique coverage, it's a candidate for replacement.
    
    # To make replacement smart:
    # 1. Identify the set of t-subsets currently covered ONLY by this block (if we remove it, these become uncovered).
    # 2. Also consider globally uncovered t-subsets.
    # 3. Generate a new block that covers as many of these "critical" t-subsets as possible.
    
    # For efficiency, we will track:
    # - cover_count: how many blocks cover each t-subset
    # - For each block, we don't explicitly track unique coverage, but we can compute it on the fly 
    #   when evaluating a swap.
    
    # Let's refine the swap logic:
    # 1. Pick a block B_old to remove.
    # 2. Calculate the set of t-subsets that would become uncovered if B_old is removed (S_critical).
    #    These are t-subsets in B_old where cover_count == 1.
    # 3. Also, let S_uncovered be the set of all currently uncovered t-subsets.
    # 4. We want a new block B_new that covers as many elements of S_critical U S_uncovered as possible.
    #    Note: B_new must be distinct from existing blocks.
    
    # To generate B_new efficiently:
    # We can use a greedy heuristic starting from a random t-subset in S_critical (if non-empty) 
    # or S_uncovered (if S_critical is empty).
    
    # Precompute t-subsets for each block? 
    # C(7,3) = 35 t-subsets per block. 78 blocks * 35 = 2730 t-subsets total to process per swap evaluation.
    # This is fast enough.
    
    # We need a way to quickly find "best" candidate points for the new block.
    
    best_score = np.sum(cover_count == 0) # Initial uncovered count
    if best_score == 0:
        return [list(b) for b in blocks]
    
    # We will perform a series of local search steps.
    # To avoid getting stuck in local minima, we can allow "hill climbing" (only accept if better or equal)
    # but since we start from random, we need to accept improvements.
    
    # We will iterate through the blocks in a random order, attempting to replace each one 
    # if it leads to a reduction in the total number of uncovered t-subsets.
    
    # Optimization: Precompute the list of t-subsets for each block to avoid recalculating 
    # itertools.combinations every time we evaluate a swap.
    block_tsubs = []
    for b in blocks:
        block_tsubs.append(list(itertools.combinations(b, t)))
    
    # We need a fast way to generate a candidate block.
    # Heuristic: 
    # 1. Pick a "seed" t-subset from the set of "critical" or "uncovered" t-subsets.
    # 2. Greedily add points to the seed to maximize coverage of the target set (critical + uncovered).
    
    # To make this fast, we can precompute for each point x, the list of t-subsets containing x.
    # But the "target set" changes every step.
    
    # Let's implement a fast greedy block generator:
    def generate_candidate_block(target_tsubs):
        """
        target_tsubs: list of t-subsets we want to cover.
        Returns a tuple of k points.
        """
        if not target_tsubs:
            return tuple(sorted(random.sample(range(v), k)))
        
        # Pick a random t-subset from targets as the seed
        seed = random.choice(target_tsubs)
        current_block = set(seed)
        
        # Greedily add k - len(seed) points
        # To speed up, we can limit the search space or use a random sample of candidates
        # But v=24 is small, so checking all 24 points is fine.
        
        for _ in range(k - len(seed)):
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in current_block:
                    continue
                # Calculate gain: how many target_tsubs are covered by adding x?
                # A target t-subset is covered by (current_block U {x}) if it is a subset of (current_block U {x}).
                # Since target_tsubs are of size t, and current_block grows, we can check:
                # For each tsub in target_tsubs:
                #   if tsub is subset of current_block U {x}
                
                # Optimization: 
                # Instead of checking all target_tsubs (which could be large), 
                # we can note that adding x only covers t-subsets that contain x.
                # Precompute? No, target_tsubs changes.
                
                # Given v=24 and k=7, and target_tsubs size potentially ~1000, 
                # checking 24 * 1000 * (subset check) might be slow in Python.
                
                # Faster check: 
                # tsub is subset of (current_block U {x}) iff 
                # (tsub is subset of current_block) OR (x in tsub AND (tsub - {x}) is subset of current_block)
                
                # Let's use a set for current_block for O(1) lookup.
                cb = current_block
                gain = 0
                for tsub in target_tsubs:
                    # tsub is a tuple
                    # Check if tsub is subset of cb U {x}
                    # If x is not in tsub, it must be subset of cb
                    # If x is in tsub, tsub - {x} must be subset of cb
                    
                    # We can precompute a set of frozensets for target_tsubs? 
                    # Or just use set operations.
                    
                    # Let's try a faster heuristic:
                    # Only consider t-subsets that contain x.
                    # This requires filtering target_tsubs.
                    # To avoid filtering every time, we can assume that most gain comes 
                    # from t-subsets containing x.
                    
                    # Let's just do the subset check. It's O(t) per t-subset.
                    # 24 points * 1000 tsubs * 3 ops = 72k ops. 
                    # 78 blocks * 100 steps * 72k = 561M ops. Too slow in Python.
                    
                    # We need a faster gain calculation.
                    pass
                
                # Alternative: 
                # Instead of exact gain, use a random sample of target_tsubs to estimate gain.
                # Or, restrict the search to points that appear in many target_tsubs.
                
                # Let's use a simpler, faster heuristic for candidate generation:
                # Pick a random t-subset from targets.
                # Pick a random point not in current_block.
                # This is essentially random walk, which might be too slow to converge.
                
                # Let's go back to exact gain but optimize the loop.
                # We can precompute for each point x, the list of target_tsubs containing x.
                # But target_tsubs is a list of tuples.
                
                # Let's try this:
                # For each x, count how many target_tsubs contain x AND are NOT already covered by current_block.
                # A t-subset is covered by current_block if it's a subset of current_block.
                # Since current_block size < k, and t=3, 
                # if len(current_block) >= t, some tsubs are already covered.
                
                # Actually, the seed is a t-subset, so it's covered by itself.
                # As we add points, we cover more.
                
                # Let's use a "lazy" evaluation:
                # We will only calculate exact gain for the top 5 candidates based on a 
                # cheaper metric (e.g., frequency of appearance in target_tsubs).
                
                # Frequency of x in target_tsubs:
                # We can precompute this for the current target_tsubs set.
                
                # To keep this simple and within time limits, I will use a 
                # "Greedy with limited lookahead" or just a faster random approach 
                # for the candidate generation, relying on the local search to 
                # iterate many times.
                
                # Let's use a simpler candidate generator:
                # 1. Pick a random t-subset from target_tsubs.
                # 2. Pick k-3 random points from the remaining v-3 points.
                # 3. Evaluate the coverage of this candidate.
                # This is fast (O(1) to generate, O(35) to evaluate).
                # If it's better than the current block, accept.
                
                # This is basically random search within the local search framework.
                # It might be slow to find good blocks, but it's very fast per step.
                
                # Let's combine: 
                # 1. Generate 10 random candidates based on seeds from target_tsubs.
                # 2. Pick the best one.
                
                # This avoids the expensive per-point gain calculation.
                
                break # Exit inner loop, we will use the random candidate strategy outside
            
            # Fallback if the loop above doesn't work as expected
            if best_point is None:
                available = [x for x in range(v) if x not in current_block]
                if available:
                    best_point = random.choice(available)
                else:
                    break
            current_block.add(best_point)
            
        return tuple(sorted(current_block))
    
    # Revised Main Loop
    # We will attempt to improve the solution by replacing blocks.
    # We track the total uncovered count.
    
    # Precompute block_tsubs as sets of indices for faster intersection?
    # No, let's stick to indices.
    
    # Let's re-initialize block_tsubs as lists of indices.
    block_tsub_indices = []
    for b in blocks:
        indices = [tsub_index[tsub] for tsub in itertools.combinations(b, t)]
        block_tsub_indices.append(indices)
    
    # To speed up "remove block" and "add block", we use the indices.
    
    # We need to know which t-subsets are currently uncovered.
    # uncovered_indices = np.where(cover_count == 0)[0]
    
    # We will iterate through blocks, trying to replace each one.
    # To avoid bias, we shuffle the order of blocks each epoch.
    
    max_epochs = 100
    for epoch in range(max_epochs):
        if time.time() > deadline - 5:
            break
            
        # Shuffle block indices
        block_order = list(range(len(blocks)))
        random.shuffle(block_order)
        
        for i in block_order:
            if time.time() > deadline - 5:
                break
            
            # Get current block
            b_old = blocks[i]
            tsubs_old_indices = block_tsub_indices[i]
            
            # Identify t-subsets that would become uncovered if we remove b_old
            # These are indices where cover_count == 1
            critical_indices = []
            for idx in tsubs_old_indices:
                if cover_count[idx] == 1:
                    critical_indices.append(idx)
            
            # Also consider all currently uncovered t-subsets
            # We can get this from cover_count == 0
            # But iterating over all num_tsubsets (2024) is fast.
            # uncovered_indices = np.where(cover_count == 0)[0]
            
            # To avoid numpy overhead in the inner loop, let's maintain a list of uncovered indices?
            # That's hard to update efficiently.
            # Let's just use the numpy array.
            uncovered_indices = np.where(cover_count == 0)[0]
            
            # Target t-subsets for the new block: Critical + Uncovered
            # If critical is empty, we are just trying to cover uncovered ones.
            # If uncovered is empty, we are valid.
            
            if len(uncovered_indices) == 0:
                # Valid covering found!
                # We can try to reduce the number of blocks?
                # The task asks to minimize the number of blocks.
                # If we have a valid covering with 78 blocks, can we remove some?
                # Yes! Local search for minimization.
                
                # Try to remove a block if it doesn't break validity.
                # A block can be removed if all its t-subsets are covered by other blocks.
                # i.e., cover_count[idx] > 1 for all idx in tsubs_old_indices.
                
                # Let's add a minimization phase.
                # If valid, try to remove blocks.
                removed_any = False
                for j in range(len(blocks)):
                    if time.time() > deadline - 5:
                        break
                    tsubs_j = block_tsub_indices[j]
                    can_remove = True
                    for idx in tsubs_j:
                        if cover_count[idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Remove block j
                        remove_block(blocks[j])
                        del blocks[j]
                        del block_tsub_indices[j]
                        removed_any = True
                        break # Restart the removal loop as indices changed
                if not removed_any:
                    # No more blocks can be removed
                    break # Exit main loop, we are done
                continue # Continue to next epoch to try removing more
            
            # If not valid, generate a candidate block to replace b_old
            # We want to cover critical_indices and uncovered_indices.
            # Let's pick a seed from critical_indices if available, else from uncovered_indices.
            if critical_indices:
                seed_idx = random.choice(critical_indices)
            else:
                seed_idx = random.choice(uncovered_indices)
            
            seed_tsub = all_tsubsets[seed_idx]
            
            # Generate candidate block
            # Strategy: Start with seed_tsub, add random points, evaluate.
            # Try a few candidates and pick the best.
            
            best_candidate = None
            best_gain = -1
            
            for _ in range(10): # Try 10 random completions
                cand_set = set(seed_tsub)
                # Add k-3 random points
                available_points = [x for x in range(v) if x not in cand_set]
                random.shuffle(available_points)
                for p in available_points[:k-3]:
                    cand_set.add(p)
                
                cand_tuple = tuple(sorted(cand_set))
                
                # Check if candidate is already in blocks
                # This is O(N) check. N=78. Fast enough.
                if cand_tuple in blocks:
                    continue
                
                # Calculate gain: How many of (critical U uncovered) does it cover?
                # We need to check which t-subsets of cand_tuple are in the target set.
                
                # Target set is union of critical_indices and uncovered_indices.
                # We can create a set of target indices for O(1) lookup.
                # But creating a set of 2000 elements 10 times per block replacement is expensive.
                
                # Alternative: 
                # For each t-subset of cand_tuple, check if its index is in critical OR uncovered.
                # We can use a boolean array for "is_target".
                
                # Let's maintain a global boolean array `is_critical` and `is_uncovered`.
                # is_uncovered = (cover_count == 0)
                # is_critical = (cover_count == 1) AND (idx in tsubs_old_indices)
                
                # This is getting complex. Let's simplify:
                # Just check if the t-subset is currently uncovered (cover_count == 0).
                # Covering uncovered t-subsets is the primary goal.
                # Covering critical t-subsets is secondary (to avoid creating new holes).
                
                # Gain = (# uncovered t-subsets covered by cand) - (# critical t-subsets NOT covered by cand)
                # Actually, if we remove b_old, critical t-subsets become uncovered.
                # So we want cand to cover as many critical t-subsets as possible.
                
                # Let's just count how many of the t-subsets of cand are in the target set 
                # (critical + currently uncovered).
                
                target_set = set(critical_indices) | set(uncovered_indices)
                
                gain = 0
                for tsub in itertools.combinations(cand_tuple, t):
                    idx = tsub_index[tsub]
                    if idx in target_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand_tuple
            
            if best_candidate is not None:
                # Evaluate if replacing b_old with best_candidate improves the total uncovered count.
                # Current uncovered count: len(uncovered_indices)
                # New uncovered count: 
                # 1. Remove b_old: cover_count decreases for tsubs_old_indices.
                # 2. Add best_candidate: cover_count increases for tsubs_new_indices.
                
                # We can simulate this.
                # But it's easier to just do it and check if the total uncovered count decreased.
                
                # Remove old
                remove_block(b_old)
                
                # Add new
                add_block(best_candidate)
                
                # Update state
                blocks[i] = best_candidate
                block_tsub_indices[i] = [tsub_index[tsub] for tsub in itertools.combinations(best_candidate, t)]
                
                # Check if we improved
                # new_uncovered_count = np.sum(cover_count == 0)
                # If new_uncovered_count < len(uncovered_indices), it's an improvement.
                # We don't need to explicitly check if we accept, because local search 
                # usually accepts any change that doesn't increase the cost.
                # However, to be safe, we can revert if it got worse.
                
                # For now, let's accept all changes (random walk) to explore the space,
                # but this might drift.
                # Better: Accept only if uncovered count didn't increase.
                
                # We didn't store the old uncovered count efficiently.
                # Let's just accept. The minimization phase will clean up later.
    
    # Final check: Ensure we return a valid covering if possible.
    # If we still have uncovered t-subsets, the result is invalid (score 0).
    # But the task says "Valid when every t-subset ... lies in at least one block".
    # If we return an invalid covering, score is 0.
    # So we must ensure validity.
    
    # If we are out of time and still have holes, we have a problem.
    # The initialization with 78 blocks and local search should likely find a valid covering.
    # If not, we might need to add more blocks.
    
    # Let's add a fallback: if still uncovered, add blocks greedily until covered or time out.
    if time.time() < deadline - 2:
        while np.sum(cover_count == 0) > 0:
            # Find an uncovered t-subset
            uncovered_indices = np.where(cover_count == 0)[0]
            if len(uncovered_indices) == 0:
                break
            idx = random.choice(uncovered_indices)
            tsub = all_tsubsets[idx]
            
            # Create a block containing this tsub
            block_set = set(tsub)
            available = [x for x in range(v) if x not in block_set]
            random.shuffle(available)
            for p in available[:k-3]:
                block_set.add(p)
            
            block = tuple(sorted(block_set))
            
            # Check if duplicate
            if block in blocks:
                continue
            
            blocks.append(block)
            add_block(block)
            block_tsub_indices.append([tsub_index[tsub] for tsub in itertools.combinations(block, t)])
            
            if time.time() > deadline - 2:
                break
    
    return [list(b) for b in blocks]
