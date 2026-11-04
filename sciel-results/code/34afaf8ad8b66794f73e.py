import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: Evaluator limit is 150s. We stop at 140s to allow for cleanup/return.
    deadline = start_time + 140.0
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map t-subset tuple to index for fast lookup
    ts_idx_map = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block:
    # - The list of t-subset indices it covers
    # - A bitmask of those t-subset indices
    block_ts_indices = []
    block_masks = []
    
    # Inverse index: ts_index -> list of block indices that cover it
    ts_to_blocks = [[] for _ in range(n_tsubsets)]
    
    for b_idx, blk in enumerate(all_blocks):
        covered_indices = []
        mask = 0
        # Find all t-subsets inside this block
        for ts in itertools.combinations(blk, t):
            idx = ts_idx_map[ts]
            covered_indices.append(idx)
            mask |= (1 << idx)
            ts_to_blocks[idx].append(b_idx)
        block_ts_indices.append(covered_indices)
        block_masks.append(mask)
        
    # Helper to get current time
    def get_time():
        return time.time()
        
    # Greedy construction with pivot
    def run_greedy(seed_val):
        random.seed(seed_val)
        
        # remaining_mask: bit i is 1 if t-subset i is NOT yet covered
        remaining_mask = (1 << n_tsubsets) - 1
        selected_blocks = []
        
        # We need to track counts for the reduction phase later
        # But for greedy, we just need to know what's covered.
        # To save memory/time, we can just use the mask.
        # However, reduction needs to know if a block is redundant (all its ts have count > 1).
        # We can reconstruct counts from the final list of blocks if needed, 
        # or maintain them. Maintaining counts is O(1) per block update if we iterate covered ts.
        
        # Let's maintain counts for the reduction phase to avoid re-scanning everything at the end.
        # Actually, re-scanning is fine for 300-400 blocks. 
        # Let's stick to just the mask for speed during greedy.
        
        while remaining_mask:
            if get_time() > deadline:
                break
            
            # Pick a random uncovered t-subset as pivot
            # To pick a random set bit, we can use a trick or just iterate.
            # Since n_tsubsets is 11628, we can't just use random.choice on a list of bits easily.
            # We can extract the bits into a list? No, that's O(N).
            
            # Better: 
            # 1. If remaining_mask is small, iterate.
            # 2. Use a random shift? 
            #    Let's just pick a random index in range(n_tsubsets) and check if it's set.
            #    Expected iterations is small if coverage is high.
            
            pivot_ts_idx = -1
            # Optimization: If the number of remaining bits is small, we can find them faster?
            # For now, random sampling is usually fast enough.
            # But we must ensure we find one.
            
            # A more robust way to pick a random set bit:
            # We can maintain a list of remaining indices? Updating it is O(N).
            # Let's stick to random sampling with a limit.
            
            found = False
            # Try up to 100 random indices
            for _ in range(100):
                r_idx = random.randint(0, n_tsubsets - 1)
                if (remaining_mask >> r_idx) & 1:
                    pivot_ts_idx = r_idx
                    found = True
                    break
            
            if not found:
                # Fallback: linear scan (should be rare)
                for i in range(n_tsubsets):
                    if (remaining_mask >> i) & 1:
                        pivot_ts_idx = i
                        break
            
            if pivot_ts_idx == -1:
                # Should not happen if remaining_mask != 0
                break
            
            # Find the best block covering pivot_ts_idx
            best_block_idx = -1
            best_gain = -1
            
            candidates = ts_to_blocks[pivot_ts_idx]
            
            for b_idx in candidates:
                # Gain is the number of remaining t-subsets covered by this block
                # mask & remaining_mask gives the bits covered by b that are still remaining
                gain_mask = block_masks[b_idx] & remaining_mask
                gain = gain_mask.bit_count() # Python 3.10+
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    # Early exit if we cover all remaining?
                    # If gain == popcount(remaining_mask), we are done after this.
                    # But we don't know popcount(remaining_mask) without calculating it.
                    # Calculating bit_count of remaining_mask is O(N/word_size).
                    # Let's skip early exit for simplicity, or check if gain == n_tsubsets? No.
            
            if best_block_idx == -1:
                # This shouldn't happen if the problem is solvable
                break
                
            selected_blocks.append(best_block_idx)
            
            # Update remaining mask
            remaining_mask &= ~block_masks[best_block_idx]
            
        return selected_blocks

    def reduce_blocks(block_indices):
        if not block_indices:
            return []
            
        # Calculate coverage counts for each t-subset
        counts = [0] * n_tsubsets
        for b_idx in block_indices:
            for ts_idx in block_ts_indices[b_idx]:
                counts[ts_idx] += 1
        
        # Iteratively remove redundant blocks
        # A block is redundant if all t-subsets it covers have count > 1
        # We process blocks in reverse order of selection (greedy often picks "large" blocks first, 
        # so later blocks might be more specialized/redundant? Or earlier ones? 
        # Actually, greedy picks max gain. Later picks have lower gain. 
        # It's often better to try removing later blocks first, or just iterate all).
        
        # To be safe and effective, we iterate through the list and try to remove.
        # We need to be careful with indices if we remove from the list.
        # We'll use a "to_remove" set or build a new list.
        
        kept_blocks = []
        # We need to re-evaluate redundancy based on the *current* set of kept blocks?
        # No, standard reduction: 
        # If we remove a block, counts decrease. A previously non-redundant block might become redundant?
        # No, if a block was NOT redundant (had a unique ts), removing another block doesn't make that ts unique again (count decreases).
        # Wait. If block A covers ts X (count 2: A and B). Block B covers ts Y (count 2: A and B).
        # If we remove A, count of X becomes 1 (covered by B). Count of Y becomes 1 (covered by B).
        # B is still not redundant? 
        # Actually, redundancy is defined relative to the *current* set.
        # If we remove blocks one by one, we must re-check.
        
        # Efficient Reduction Strategy:
        # 1. Start with all blocks.
        # 2. Loop:
        #    - Find a block that is redundant (all its ts have count > 1).
        #    - Remove it.
        #    - Update counts.
        #    - Repeat until no redundant blocks found.
        
        # To speed this up, we can just do a single pass in reverse?
        # Often, a single pass in reverse is effective for greedy solutions.
        # Let's try a single pass first, and if time permits, iterate.
        
        # Let's do a robust iterative reduction but with a time check.
        current_blocks = list(block_indices)
        
        # Pre-compute ts lists for current blocks? No, use global.
        
        changed = True
        # Limit iterations to prevent infinite loops or excessive time
        max_iterations = len(current_blocks) * 2
        
        while changed and max_iterations > 0:
            changed = False
            max_iterations -= 1
            
            # We scan from end to start (later greedy picks are often less critical)
            for i in range(len(current_blocks) - 1, -1, -1):
                if get_time() > deadline:
                    return current_blocks
                    
                b_idx = current_blocks[i]
                is_redundant = True
                
                for ts_idx in block_ts_indices[b_idx]:
                    if counts[ts_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block
                    current_blocks.pop(i)
                    # Update counts
                    for ts_idx in block_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    changed = True
                    # Restart scan? Or continue?
                    # Continuing is safer for time, but might miss new redundancies created by this removal?
                    # Removal only decreases counts. If a block was not redundant (had count 1 ts), 
                    # it won't become redundant. 
                    # So we don't need to restart. We can continue scanning.
                    # However, if we remove block i, the indices of blocks after i shift? 
                    # No, we are iterating backwards. Popping i doesn't affect indices < i.
                    # So we can just continue the loop.
                    
        return current_blocks

    best_solution = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Estimate time per restart:
    # Greedy: ~350 steps. Each step scans ~364 candidates. 350 * 364 = 127k ops.
    # Python overhead: ~1-2 ms per step? So ~0.5s per restart?
    # If 0.5s/restart, we can do ~200 restarts in 100s.
    # Let's try to run as many as possible.
    
    seed_offset = 42
    restart_count = 0
    
    while True:
        if get_time() > deadline - 5.0: # Leave 5s for final reduction/return
            break
            
        seed = seed_offset + restart_count
        sol = run_greedy(seed)
        restart_count += 1
        
        if not sol:
            continue
            
        # Reduction
        reduced_sol = reduce_blocks(sol)
        
        if len(reduced_sol) < best_len:
            best_len = len(reduced_sol)
            best_solution = reduced_sol[:]
            
        # If we get a very good solution, we might want to spend more time polishing?
        # But the prompt says "minimize number of blocks". 
        # If we hit 335 (best known), we are doing well.
        # If we hit < 335, that's a breakthrough.
        
        # Heuristic: if we've found a solution <= 335, we can keep searching to see if we can go lower.
        # No early stop unless time runs out.
        
    # Fallback if no solution found (should not happen)
    if best_solution is None:
        # Run one deterministic greedy
        sol = run_greedy(0)
        best_solution = reduce_blocks(sol)
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_solution]
    
    return final_blocks

