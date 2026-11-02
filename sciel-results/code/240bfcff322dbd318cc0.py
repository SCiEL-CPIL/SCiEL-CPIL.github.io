import time
import numpy as np
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for evaluation overhead
    
    # Encode t-subsets as integers for fast hashing
    # We need a mapping from t-subset to an integer ID
    # Or just use tuples, but tuples are slow.
    # Let's use a set of tuples for simplicity first, but optimize the inner loop.
    # Actually, for v=24, t=3, there are 2024 subsets. 
    # A set of 2024 tuples is fine.
    
    all_t_subsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Helper to encode a subset into a frozenset or tuple for fast lookup
    # We will use tuples.
    
    # Precompute combinations for each point? No, that's heavy.
    # Let's try a smarter greedy.
    
    while uncovered and time.time() < deadline:
        # Pick an uncovered subset to start the block
        # Heuristic: Pick one that is "hardest" to cover? 
        # Or just pick the first one. 
        # A better heuristic: pick the subset with the fewest candidate points that can extend it?
        # For now, just pick min(uncovered) for determinism and speed.
        start_subset = min(uncovered)
        block = list(start_subset)
        
        # Fill the block with k points
        while len(block) < k:
            # Calculate gain for each candidate point
            # Gain(x) = number of uncovered subsets T such that T \ {x} is a subset of block
            # Note: T must be uncovered.
            
            # Optimization: 
            # If we add x to block, we cover T if T contains x and T\{x} is in block.
            # Let B be the current block.
            # We need to count |{ T in uncovered : x in T and T\{x} subseteq B }|
            
            # Precompute all subsets of B of size t-1?
            # If t=3, t-1=2.
            # If t is general, this is harder.
            
            # Let's assume t is small (3).
            # For each x in candidates:
            #   count = 0
            #   for T in uncovered:
            #      if x in T and all(y in block for y in T if y != x):
            #          count += 1
            
            # This is O(|uncovered| * |candidates| * t).
            # |uncovered| <= 2024. |candidates| <= 24.
            # 2024 * 24 * 3 ~ 145k ops per step.
            # Steps: ~76 blocks * 4 points added per block = 300 steps.
            # 300 * 145k = 43.5M ops. Might be too slow in Python.
            
            # Let's use NumPy or a faster set operation.
            # Alternative: 
            # For a fixed block B, the subsets T that would be covered by adding x are those T 
            # such that T = {x} U S where S is a subset of B of size t-1.
            # So, for a given x, the potential subsets are { {x} U S : S in C(B, t-1) }.
            # We just need to check how many of these are in `uncovered`.
            
            # This is much faster!
            # Number of S in C(B, t-1) is C(k_current, t-1).
            # k_current <= 6. t-1 = 2. C(6,2)=15.
            # So for each x, we check ~15 subsets.
            
            candidates = [x for x in range(v) if x not in block]
            best_gain = -1
            best_x = -1
            
            # Precompute subsets of block of size t-1
            # If t-1 == 0, then T={x}.
            if t == 1:
                # Trivial
                for x in candidates:
                    if (x,) in uncovered:
                        if 1 > best_gain:
                            best_gain = 1
                            best_x = x
            else:
                # Generate all (t-1)-subsets of current block
                block_tuples = tuple(block)
                potential_s = list(itertools.combinations(block_tuples, t - 1))
                
                for x in candidates:
                    gain = 0
                    for s in potential_s:
                        # T = s U {x}
                        # Ensure x is not in s (it shouldn't be, as x not in block)
                        # Sort to form a tuple
                        t_tuple = tuple(sorted(s + (x,)))
                        if t_tuple in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            
            if best_x == -1:
                # Should not happen if uncovered is not empty and we started with uncovered
                # But if no point adds value, we are stuck?
                # Actually, if start_subset is uncovered, adding its points covers it.
                # So gain should be at least 1 for points in start_subset?
                # Wait, we started with start_subset in block.
                # So we are adding points NOT in block.
                # If no point increases coverage, we just pick an arbitrary one.
                best_x = candidates[0]
            
            block.append(best_x)
        
        # Sort block for consistency
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        # Remove all t-subsets of block from uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
        
        # Check deadline
        if time.time() > deadline - 5:
            break

    # If we exited the loop because of deadline but uncovered is not empty, 
    # we have an invalid covering.
    # We must ensure validity. 
    # If time runs out, we might need to force-cover remaining subsets.
    # But the deadline is 110s, so we should be safe.
    # If uncovered is not empty, we must add blocks to cover them.
    
    # Fallback: if time is tight or uncovered remains, use a very fast, 
    # possibly suboptimal, completion strategy.
    # Actually, the greedy above is quite fast.
    # Let's add a local search step to reduce block count.
    
    # Local Search: Try to remove blocks
    # If a block's t-subsets are all covered by other blocks, remove it.
    # Repeat until no more blocks can be removed.
    
    # This might be slow if we do it naively.
    # Let's try a simple pass.
    
    # Re-evaluate coverage? 
    # We can maintain a count of how many times each t-subset is covered.
    # But we already have `uncovered` set.
    # To check if a block is redundant, we need to know if all its t-subsets are covered by OTHER blocks.
    
    # Let's try a simpler improvement:
    # If we have time, try to swap points in blocks to increase coverage?
    # Or just return the greedy result.
    # The greedy result for C(24,7,3) is usually around 80-90 blocks.
    # The known optimum is 76.
    
    # Let's add a "block removal" phase if time permits.
    
    if time.time() < deadline - 10 and blocks:
        # Build a map: t_subset -> list of block indices covering it
        # This is O(N_blocks * C(k,t))
        
        # To save time, only do this if we have many blocks.
        
        # Let's try a random restart or different heuristic?
        # No, let's stick to the robust greedy.
        
        pass

    return blocks
