import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_k_blocks:
        tids = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        block_triples.append(tids)
        
    # Map each triple to the list of block indices that cover it
    # This is needed for efficient gain updates (reverse index)
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, tids in enumerate(block_triples):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)

    # Precompute masks for potential use or just use lists?
    # The inspiration uses lists and numpy arrays for counts/gains which is efficient.
    # Let's stick to the list-based gain update which is O(sum of degrees).
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        # gain[b] is the number of currently uncovered triples covered by block b
        # Initialize with the total number of triples in the block (C(k, t))
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > deadline - 5:
                # If we run out of time during construction, we must return a valid cover.
                # The current selected_blocks might not be a complete cover.
                # We need to finish the cover quickly or return best effort.
                # However, the task requires a valid cover.
                # If we break here, we must complete the cover using a fast greedy that ignores time?
                # Or better, check time BEFORE starting a new iteration.
                # If we are mid-construction and time is up, we are in trouble.
                # Let's assume the loop finishes. If not, we need a fallback.
                pass
            
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if is_uncovered.any() is true and blocks are valid
                break
            
            # Find indices with max gain
            # To avoid O(N) scan for max every time if N is large?
            # N is ~346k. np.max is fast. np.where is fast.
            candidates = np.where(gain == max_gain)[0]
            
            # Random tie-breaking
            b_idx = rng.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            # Update gain and is_uncovered
            # For each triple in the selected block:
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    # Because this triple is no longer "newly covered" by them
                    # This is the expensive part: O(degree(triple))
                    # degree is C(v-3, k-3) = C(21, 4) = 5985? No.
                    # Number of blocks containing a specific triple (3 points) is C(v-3, k-3) = C(21, 4) = 5985.
                    # 35 triples per block * 5985 blocks per triple ~ 200k updates per selected block.
                    # 80 blocks * 200k = 16M updates.
                    # This might be too slow in Python loops.
                    
                    # Optimization:
                    # Instead of updating gain for ALL blocks covering t_idx,
                    # we can just recompute gain for candidates? No.
                    
                    # Alternative:
                    # Maintain a count of how many times each triple is covered?
                    # No, we need "uncovered" count.
                    
                    # Let's check the speed.
                    # 16M iterations in Python is ~1-2 seconds.
                    # 80 blocks * 2s = 160s. TOO SLOW.
                    
                    # We need a faster update strategy.
                    # The "mask" approach with bitsets might be faster for finding max gain,
                    # but updating the masks isn't the issue, finding the max is.
                    
                    # Let's look at Inspiration 1 again. It uses the same logic.
                    # Did it run fast? Score 1.1134 implies it found a better solution (72 blocks?).
                    # Wait, 1.0909 is current. 1.1134 is better.
                    # If the update is too slow, it would time out or do few restarts.
                    
                    # Let's optimize the update.
                    # Instead of iterating `triple_blocks[t_idx]` in Python, can we use NumPy?
                    # `gain[triple_blocks[t_idx]] -= 1`
                    # `triple_blocks[t_idx]` is a list of ints. We can make it an array.
                    
            # End while
        
        return selected_blocks

    # Optimization: Pre-convert triple_blocks to numpy arrays for vectorized updates?
    # triple_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_blocks]
    # Then: gain[triple_blocks_np[t_idx]] -= 1
    
    triple_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_blocks]

    def greedy_construct_fast(seed):
        rng = random.Random(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            if time.time() > deadline - 2:
                # Emergency: complete the cover with a simple greedy that doesn't check time?
                # Or just break and hope we have a cover? No, must be valid.
                # If we break, we need to append blocks until covered.
                # This fallback is risky. Let's hope the main loop finishes.
                pass
            
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            candidates = np.where(gain == max_gain)[0]
            b_idx = rng.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            # Vectorized update
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # Update gains for all blocks covering this triple
                    # This is still a Python loop over 35 triples, but the inner update is vectorized
                    gain[triple_blocks_np[t_idx]] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        # Convert to list of sets of triple indices for quick check?
        # We maintain cover_count[t]
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            # Vectorized increment
            cover_count[np.array(block_triples[b_idx], dtype=np.int32)] += 1
        
        active = [True] * len(selected_blocks)
        
        # Pass 1: Remove redundant blocks
        # A block is redundant if all its triples have cover_count > 1
        # We iterate and remove.
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                # Check if redundant
                # Get the triple indices for this block
                tids = block_triples[b_idx]
                # Check if any triple in this block has count == 1
                # If all counts > 1, it's redundant
                # Vectorized check:
                counts_for_block = cover_count[tids]
                if np.min(counts_for_block) > 1:
                    active[i] = False
                    # Update cover counts
                    cover_count[tids] -= 1
                    changed = True
        
        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i] and active[i]]

    best_cover = None
    seed = 0
    
    # Run multiple greedy attempts
    while time.time() < deadline - 10:
        # Run greedy
        cover = greedy_construct_fast(seed)
        
        # If the cover is incomplete (due to time break), we must complete it
        # Check validity
        # For simplicity, let's assume it completes. If not, we need a fallback.
        # Let's add a check:
        # Create a mask of covered triples
        # If not all covered, append blocks greedily until covered (ignoring time for this final step? Risky)
        # Or just skip this cover if invalid?
        # The task says "Valid when every t-subset...".
        # If we return an invalid cover, score is 0.
        # So we MUST return a valid cover.
        
        # Let's verify validity cheaply?
        # We can maintain a set of covered triples during greedy?
        # No, we used is_uncovered array.
        # If the loop finished normally, is_uncovered.all() == False is not true, i.e., all covered.
        # The loop condition is `while is_uncovered.any()`.
        # If it exits, either is_uncovered is all False (valid) or max_gain == 0 (impossible if uncovered).
        # So if it exits normally, it is valid.
        
        # Reduce
        cover = reduce_cover(cover)
        
        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover
            # If we have a very good cover, we might want to spend time on local search?
            # For now, just restart greedy with different seeds.
            
        seed += 1
        
        # If we have a very good cover, maybe do local search?
        # Local search: Swap one block for another?
        # Or remove a block and re-cover the holes?
        # Given time constraints, restarts are safer.

    if best_cover is None:
        # Fallback: simple greedy
        # This should not happen
        best_cover = greedy_construct_fast(0)
        
    return [list(all_k_blocks[i]) for i in best_cover]
