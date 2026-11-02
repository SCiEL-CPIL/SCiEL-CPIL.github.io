import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all t-subsets (triples for t=3)
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all possible k-blocks and their covered subset indices
    # For v=24, k=7, this is C(24,7) = 346,104 blocks.
    # Each block covers C(7,3) = 35 subsets.
    # Storing this is ~346k * 35 ints ~ 12M ints, which is fine.
    
    # To save memory/time in precomputation, we can generate blocks on the fly 
    # or precompute a list of (block_tuple, list_of_subset_indices).
    # Given the 120s limit, precomputing 346k blocks might take a few seconds.
    # Let's estimate: 346k * 35 ops is ~12M ops. Python can do ~10M-50M simple ops/sec.
    # So precomputation might take 1-2 seconds. This is acceptable.
    
    all_blocks = []
    block_to_subsets = []
    
    # Optimization: We don't need to store the block itself if we can regenerate it,
    # but we need to return the final blocks. So we store the block tuples.
    
    # We will generate all blocks.
    # To speed up, we can use numpy? No, standard itertools is okay for 346k.
    # Let's try to be efficient.
    
    # Actually, generating all C(24,7) blocks explicitly might be memory heavy if we store lists.
    # 346,104 blocks * 7 ints = 2.4M ints.
    # 346,104 lists of subset indices (length 35) = 12M ints.
    # Total ~14M ints. In Python, an int is ~28 bytes. 14M * 28 = ~400MB.
    # This might be too high for memory limits or slow.
    
    # Alternative: Generate blocks on the fly during the greedy step?
    # Greedy step needs to evaluate "gain" for ALL candidate blocks not yet selected.
    # If we evaluate all 346k blocks at each step, and we do ~70 steps, that's 24M evaluations.
    # Each evaluation requires checking 35 subsets. 24M * 35 = 840M ops.
    # This is likely too slow in Python (10-20s per step? No, 840M simple ops is ~10-20s total maybe?
    # Actually, Python loops are slow. 840M iterations in a tight loop might take 5-10 minutes.
    # So we MUST precompute or use a faster method.
    
    # Let's try a hybrid: Precompute the mapping of block -> covered subsets using bitmasks or sets?
    # A set of 35 ints is faster for intersection/union?
    
    # Better approach for speed:
    # 1. Precompute a list of all blocks.
    # 2. For each block, store the set of subset indices it covers.
    # 3. Use a "uncovered" bitmask or set.
    # 4. In each greedy step, iterate through all blocks not yet used, calculate gain.
    
    # To make this fast enough in 120s:
    # We need the inner loop (calculating gain) to be very fast.
    # If we use Python sets: `gain = len(uncovered_set & block_subset_set)`
    # This is C-speed for set intersection.
    
    # Let's precompute:
    # 1. `all_blocks`: list of tuples (the block itself)
    # 2. `block_subs`: list of sets of subset indices for each block
    
    # Memory check:
    # 346k sets of 35 ints.
    # This is heavy.
    
    # Alternative Strategy: 
    # Instead of iterating ALL blocks, can we restrict the candidate pool?
    # The "Logarithmic Heuristic" usually works best with the full universe of sets.
    # However, for v=24, k=7, the full universe is 346k.
    # 70 steps * 346k candidates = 24M candidate evaluations.
    # If each evaluation is ~1 microsecond (optimistic for Python set intersection of size 35),
    # 24M * 1us = 24 seconds. This is feasible!
    
    # Let's implement this carefully.
    
    # Precomputation
    block_list = []
    block_covered_sets = []
    
    # We'll generate blocks in a deterministic order but maybe shuffle for restarts?
    # No, we need the full list for the greedy choice.
    
    # To save memory, we can store the subset indices as a frozenset or just a set.
    # Let's try to generate them.
    
    # Optimization: 
    # We can skip precomputation of ALL blocks if we find a good solution quickly?
    # No, greedy needs global view.
    
    # Let's try to precompute.
    
    # If memory is an issue, we can use a different representation.
    # But let's try the set approach first.
    
    # To speed up precomputation, we can use numpy?
    # No, let's stick to Python but optimize the loop.
    
    # Actually, 346k iterations in Python to build the lists is fast (<1s).
    # The bottleneck is the greedy selection loop.
    
    # Let's refine the greedy selection.
    # We maintain `uncovered` as a set of subset indices.
    # For each candidate block, we compute `len(uncovered & block_covered_sets[i])`.
    
    # We also need to avoid selecting the same block twice.
    # We can keep a `used_blocks` set of indices.
    
    # Algorithm:
    # 1. Precompute `all_blocks` and `block_covered_sets`.
    # 2. Run multiple restarts.
    #    In each restart:
    #    a. `uncovered` = set(range(n_subsets))
    #    b. `selected_blocks` = []
    #    c. `used_block_indices` = set()
    #    d. While `uncovered` is not empty:
    #       - Find block `i` (not in `used_block_indices`) that maximizes `len(uncovered & block_covered_sets[i])`.
    #       - Tie-breaking: Randomly among max gain blocks? Or lowest index?
    #         To diversify, we can add small random noise to the gain or randomly pick from top K.
    #       - Add block `i` to `selected_blocks`.
    #       - Update `uncovered`: `uncovered -= block_covered_sets[i]`
    #       - Add `i` to `used_block_indices`.
    #    e. Post-processing: Remove redundant blocks (greedy removal).
    # 3. Keep the best solution found.
    
    # To ensure diversity in restarts:
    # - We can shuffle the order of blocks considered? No, greedy picks the best.
    # - We can use a "randomized greedy": with probability p, pick a random block from the top 10% gainers.
    # - Or, we can perturb the initial state?
    # - Or, we can use different random seeds for tie-breaking.
    
    # Let's implement the core loop efficiently.
    
    # Precomputation
    # We need to map block tuple -> index? No, we just iterate indices.
    # But we need to return the actual blocks.
    
    # Let's generate the lists.
    # To speed up generation, we can use itertools.combinations.
    
    # Note: C(24,7) is 346,104.
    # Generating 346k combinations and 35 subsets for each is 12M combinations.
    # This might take 5-10 seconds in Python.
    # Is there a faster way?
    # We can precompute the subset indices for all possible 7-subsets?
    # No, just compute on the fly.
    
    # Let's check if we can avoid precomputing ALL blocks.
    # If we use a heuristic that doesn't require global search, we could be faster.
    # But the prompt asks for a MATERIAL change. "Iterative Greedy with Log Heuristic" is a standard strong baseline for Set Cover.
    
    # Let's try to make the precomputation as fast as possible.
    
    # Alternative:
    # Use a bitmask for `uncovered`?
    # n_subsets = 2024.
    # A Python int can hold 2024 bits easily.
    # `uncovered_mask`
    # `block_mask[i]` = bitmask of subsets covered by block i.
    # `gain = bin(uncovered_mask & block_mask[i]).count('1')`
    # `bin()` and `count()` are C-speed.
    # This is MUCH faster than set intersection in Python?
    # `bin(x).count('1')` creates a string. 
    # `int.bit_count()` is available in Python 3.10+.
    # Let's assume Python 3.10+ is available (standard in modern environments).
    # If not, `bin().count('1')` is okay.
    
    # Let's use bitmasks.
    
    # Precomputation:
    # 1. `block_list`: list of tuples
    # 2. `block_masks`: list of ints (bitmasks)
    
    # Generation:
    # for block in itertools.combinations(range(v), k):
    #    mask = 0
    #    for sub in itertools.combinations(block, t):
    #        idx = subset_index[sub]
    #        mask |= (1 << idx)
    #    block_list.append(block)
    #    block_masks.append(mask)
    
    # This loop runs 346k times. Inner loop 35 times.
    # 12M iterations.
    # In Python, 12M simple ops might take ~1-2 seconds.
    # This is acceptable.
    
    # Let's implement this.
    
    # To further speed up, we can pre-compute the mask for all 3-subsets?
    # No, the mask is specific to the 7-block.
    
    # Let's write the code.
    
    # --- Precomputation ---
    block_list = []
    block_masks = []
    
    # We can parallelize? No, standard library.
    # Just a simple loop.
    
    # Optimization: 
    # We can pre-compute the index of every triple.
    # subset_index is already a dict.
    
    # Let's try to generate.
    
    # If this takes too long, we might timeout.
    # 346k * 35 = 12.1M iterations.
    # Each iteration: `sub` creation, `subset_index[sub]` lookup, `mask |= 1<<idx`.
    # This is probably ~5-10 seconds in Python.
    # Plus the greedy steps.
    # Total time budget 115s.
    # If precomputation takes 10s, we have 105s for search.
    # Search: ~70 steps * 346k candidates = 24M candidates.
    # Each candidate: `gain = (uncovered_mask & block_masks[i]).bit_count()`
    # This is very fast (C-speed).
    # 24M ops * ~10ns = 0.24s? No, Python overhead is higher.
    # Maybe ~5-10s for the whole search loop?
    # So total ~20s. We have plenty of time for multiple restarts.
    
    # Let's implement.
    
    # Handle case where v, k, t are different?
    # The task says "evaluator passes this task's parameters".
    # But the code should be general enough or at least handle the specific case well.
    # I will assume v=24, k=7, t=3 based on the prompt, but write it generically.
    # If v is larger, C(v,k) explodes.
    # For v=24, k=7, it's fine.
    # If v=30, k=7, C(30,7) = 2M. Precomputation 2M*35 = 70M ops. ~30-60s.
    # Search: 70 * 2M = 140M ops. ~10-20s.
    # Total ~60-80s. Still fits.
    # If v=35, k=7, C(35,7) = 6.7M. Precomputation 235M ops. Too slow.
    # So this strategy is only viable for small v, k.
    # Given the context (C(24,7,3)), it is viable.
    
    # Let's add a check: if C(v,k) is too large, fall back to a heuristic?
    # Or just rely on the specific task parameters.
    # The prompt implies a specific task.
    
    # Let's implement the bitmask greedy.
    
    # --- Precompute ---
    # To speed up precomputation, we can use a list for subset_index?
    # No, subset is a tuple of 3 ints. Dict is necessary.
    
    # Let's try to optimize the inner loop of precomputation.
    # `for sub in itertools.combinations(block, t):`
    # `idx = subset_index[sub]`
    # `mask |= 1 << idx`
    
    # We can pre-compute the bit for each triple?
    # `triple_bit = {triple: 1 << i for i, triple in enumerate(all_subsets)}`
    # Then `mask |= triple_bit[sub]`
    # This saves the shift operation.
    
    triple_bit = {}
    for i, sub in enumerate(all_subsets):
        triple_bit[sub] = 1 << i
        
    # Now generate blocks
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= triple_bit[sub]
        block_list.append(block)
        block_masks.append(mask)
        
    n_blocks = len(block_list)
    
    # --- Search ---
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts
    # We have ~100s left.
    # Each restart takes ~ (steps * n_blocks * cost_per_eval)
    # Let's estimate time for one restart.
    # If it's ~1s, we can do 50 restarts.
    # If it's ~10s, we can do 5.
    
    # We will run restarts until deadline.
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0:
        restart_start = time.time()
        
        uncovered_mask = (1 << n_subsets) - 1
        selected_indices = []
        used_indices = set()
        
        # Greedy Loop
        while uncovered_mask:
            best_gain = 0
            best_candidates = []
            
            # Iterate through all blocks
            # Optimization: Skip used blocks
            for i in range(n_blocks):
                if i in used_indices:
                    continue
                # Calculate gain
                # `uncovered_mask & block_masks[i]`
                # `bit_count()`
                gain = (uncovered_mask & block_masks[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                # Should not happen if a covering exists
                break
                
            # Tie-breaking: Randomly pick one
            # To add diversity, we can pick randomly from the best candidates.
            chosen_idx = rng.choice(best_candidates)
            
            selected_indices.append(chosen_idx)
            used_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        # Post-processing: Remove redundant blocks
        # We have `selected_indices`.
        # We want to remove any block whose removal leaves all triples covered.
        # This is a "greedy deletion" or "reverse greedy".
        
        # To do this efficiently, we can recompute coverage counts?
        # Or just check if the block is the ONLY one covering any triple.
        
        # Let's build a list of blocks.
        current_blocks = [block_list[i] for i in selected_indices]
        
        # To check redundancy, we can use the bitmask again?
        # It's easier to just re-run a greedy set cover on the selected blocks?
        # No, that might remove too much.
        
        # Standard approach:
        # Iterate through selected blocks.
        # For each block, check if it covers any triple that is NOT covered by any other selected block.
        # If not, remove it.
        
        # To do this fast:
        # 1. Compute `coverage_count` for each triple?
        #    We can use an array of size n_subsets.
        #    For each selected block, increment count for its triples.
        # 2. Then, for each block, check if any of its triples has count == 1.
        #    If yes, it's essential.
        #    If no, it's redundant.
        
        # Let's do this.
        
        coverage_counts = [0] * n_subsets
        
        # Map block index in selected list to its actual block?
        # We have `selected_indices` which are indices into `block_list`.
        
        for idx in selected_indices:
            mask = block_masks[idx]
            # Iterate through bits set in mask
            # Or just iterate through the 35 triples of the block?
            # Iterating through 35 triples is faster than iterating through 2024 bits.
            block = block_list[idx]
            for sub in itertools.combinations(block, t):
                coverage_counts[subset_index[sub]] += 1
                
        # Now, check for redundancy
        final_indices = []
        for idx in selected_indices:
            block = block_list[idx]
            is_essential = False
            for sub in itertools.combinations(block, t):
                if coverage_counts[subset_index[sub]] == 1:
                    is_essential = True
                    break
            if is_essential:
                final_indices.append(idx)
            else:
                # It is redundant, so we don't add it to final_indices.
                # But we should also decrement the counts for the triples it covered?
                # No, because we are just filtering.
                # Wait, if we remove a redundant block, it doesn't affect the coverage of others.
                # But if we remove multiple redundant blocks, do they become essential?
                # No, if a block was redundant (count > 1 for all its triples),
                # removing it doesn't make any other block redundant?
                # Actually, it might.
                # Example: A and B cover T. C covers T.
                # If we remove A, B and C still cover T.
                # But if we remove A and B, C is essential.
                # So we should process them in a way that accounts for this.
                
                # Better: Iterative removal.
                # But for simplicity and speed, one-pass removal is often sufficient for greedy results.
                # Or we can just keep the one-pass removal and verify at the end.
                
                pass
                
        # Let's do a more robust cleanup:
        # Re-run greedy on the final_indices? No.
        
        # Let's just use the one-pass removal for now.
        # If it's invalid, we fallback to the unremoved list.
        
        # Actually, let's just use the `selected_indices` directly if cleanup is complex.
        # The greedy set cover usually produces a near-minimal set.
        # The redundancy is usually low.
        
        # Let's try the iterative removal properly.
        # It's O(n_blocks * 35) which is fast.
        
        # We'll create a copy of coverage_counts and iterate.
        # Actually, the one-pass removal is:
        # "A set is minimal if no element can be removed."
        # Greedy deletion:
        # 1. Start with S.
        # 2. For each e in S (in some order):
        #    If S \ {e} is still a cover, remove e.
        
        # To check "S \ {e} is still a cover":
        # We need to know if e is the ONLY one covering any triple.
        # This is exactly what the `coverage_counts` check does IF we update counts.
        
        # Let's implement the iterative check properly.
        
        # We have `final_indices` from the one-pass check.
        # Let's verify if `final_indices` is a valid cover.
        
        # If it is, use it.
        # If not, use `selected_indices`.
        
        # Actually, the one-pass removal is safe if we process in reverse order of addition?
        # No.
        
        # Let's just use `selected_indices` to be safe, or implement a quick validation.
        
        # Validation:
        # Check if union of masks in `final_indices` covers all bits.
        
        union_mask = 0
        for idx in final_indices:
            union_mask |= block_masks[idx]
            
        if union_mask == (1 << n_subsets) - 1:
            current_final = final_indices
        else:
            current_final = selected_indices
            
        current_count = len(current_final)
        
        if current_count < best_count:
            best_count = current_count
            best_blocks = [block_list[i] for i in current_final]
            
        # Check time for this restart
        restart_time = time.time() - restart_start
        # If restart was very fast, we can do more.
        
    # Return the best blocks found
    if best_blocks:
        # Ensure they are lists of ints
        return [list(b) for b in best_blocks]
    else:
        # Fallback: Return a simple greedy result?
        # If the loop didn't run (e.g. precomputation took all time), we need a result.
        # But precomputation is done before the loop.
        # If the loop didn't run, it means deadline was passed during precomputation?
        # Unlikely.
        
        # If best_blocks is None, it means no valid cover was found?
        # Greedy should always find a cover if one exists.
        
        # Just return an empty list or a single block?
        # Better to return a valid cover.
        # Let's run a single fast greedy if we have time?
        # Or just return the first few blocks?
        
        # Since we have precomputed everything, let's just run one greedy if best is None.
        if time.time() < deadline - 1.0:
            uncovered_mask = (1 << n_subsets) - 1
            selected_indices = []
            used_indices = set()
            while uncovered_mask:
                best_gain = 0
                best_idx = -1
                for i in range(n_blocks):
                    if i in used_indices:
                        continue
                    gain = (uncovered_mask & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                selected_indices.append(best_idx)
                used_indices.add(best_idx)
                uncovered_mask &= ~block_masks[best_idx]
            
            if selected_indices:
                return [list(block_list[i]) for i in selected_indices]
                
        return []
