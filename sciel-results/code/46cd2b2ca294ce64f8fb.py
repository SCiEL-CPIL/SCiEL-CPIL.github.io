import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, find the indices of the t-subsets it covers
    # We represent coverage as a bitmask of tsets? No, num_tsets is 11628, too big for int64.
    # We use a boolean array or a set of indices.
    # For speed in greedy, we want to count how many *uncovered* tsets a block covers.
    # Let's precompute the list of tset indices for each block.
    
    block_tsets = []
    for b in all_blocks:
        tsets_in_b = []
        # Iterate over all t-subsets of b
        for ts in itertools.combinations(b, t):
            tsets_in_b.append(tset_to_idx[ts])
        block_tsets.append(np.array(tsets_in_b, dtype=np.int32))
        
    # Convert to a list of sets for faster intersection/difference?
    # Or just keep as arrays and use numpy for counting?
    # Since we need to update "uncovered" status, a boolean array is good.
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover(uncovered_mask, remaining_blocks_indices):
        """
        Greedily select blocks to cover uncovered tsets.
        uncovered_mask: boolean array of size num_tsets
        remaining_blocks_indices: set of indices into all_blocks that are not yet used? 
        Actually, in standard set cover, we can reuse blocks? No, blocks must be distinct.
        But we are building a list of distinct blocks.
        """
        uncovered_set = set(np.nonzero(uncovered_mask)[0])
        if not uncovered_set:
            return []
            
        blocks = []
        current_uncovered = uncovered_mask.copy()
        remaining_idx = set(range(num_blocks))
        
        while uncovered_set:
            if time.time() > deadline:
                return None # Timeout
            
            best_block_idx = -1
            best_gain = -1
            
            # Optimization: We can't check all blocks every time if it's too slow.
            # But num_blocks is 75582. num_tsets is 11628.
            # Checking gain for one block: count intersection with uncovered_set.
            # If we use a set for uncovered, intersection is O(min(len(uncovered), len(block_tsets))).
            # len(block_tsets) is C(8,5) = 56.
            # So checking one block is ~56 operations.
            # 75582 * 56 ~ 4M operations per step.
            # Number of steps? C(19,8,5) is ~335 blocks.
            # 335 * 4M = 1.3B operations. This might be too slow in Python.
            
            # Let's optimize gain calculation using NumPy if possible, or just speed up the loop.
            # Actually, we can maintain a count of how many blocks cover each tset?
            # No, we are adding blocks one by one.
            
            # Alternative: Randomized Greedy with a subset of candidates?
            # Or just accept that it might be slow and rely on the fact that we only run this once or twice?
            # The current program runs 10 attempts.
            
            # Let's try to speed up the inner loop.
            # Instead of iterating all blocks, we can pick a random uncovered tset and only consider blocks covering it?
            # That's a valid heuristic (Greedy with random tie-breaking or focus).
            
            # Pick a random uncovered tset
            target_tset = random.choice(list(uncovered_set))
            
            # Find blocks that cover this tset
            # We can precompute tset_to_blocks
            # tset_to_blocks[target_tset] gives indices of blocks covering it.
            
            # Let's precompute tset_to_blocks
            # This is the inverse of block_tsets.
            
            # Since we are inside the function, we can't easily precompute outside without passing it.
            # Let's assume we have access to a precomputed structure or build it once.
            # For now, let's stick to the simpler "check all blocks" but optimize the gain check.
            
            # Actually, let's just use the "pick random uncovered, check blocks covering it" strategy.
            # It's much faster.
            
            # We need tset_to_blocks.
            # Let's build it once outside the loop if possible, or pass it.
            # Since I can only edit the function, I will build it inside but cache it? 
            # No, I will restructure.
            
            # Let's rebuild the strategy:
            # 1. Precompute tset_to_blocks: dict mapping tset_idx -> list of block_idxs
            # 2. Greedy:
            #    - Pick random uncovered tset
            #    - For each block covering it, calculate gain (number of *other* uncovered tsets it covers)
            #    - Pick block with max gain
            #    - Add to solution, update uncovered
            
            pass # Placeholder for logic, see replacement below

    # --- RESTRUCTURED IMPLEMENTATION ---
    
    # Precompute tset_to_blocks
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            t_idx = tset_to_idx[ts]
            tset_to_blocks[t_idx].append(b_idx)
            
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = np.ones(num_tsets, dtype=bool)
        uncovered_set = set(range(num_tsets))
        blocks = []
        
        while uncovered_set:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered tset to focus on
            target_tset = random.choice(list(uncovered_set))
            
            # Get blocks covering this tset
            candidate_blocks = tset_to_blocks[target_tset]
            
            if not candidate_blocks:
                # Should not happen if problem is valid, but safety check
                break
                
            best_block = -1
            best_gain = -1
            
            for b_idx in candidate_blocks:
                # Calculate gain: how many uncovered tsets does this block cover?
                # We only care about tsets in uncovered_set
                gain = 0
                for ts_idx in block_tsets[b_idx]:
                    if uncovered[ts_idx]:
                        gain += 1
                        
                # Tie-breaking: random
                if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                    best_gain = gain
                    best_block = b_idx
                    
            if best_block == -1:
                break
                
            blocks.append(best_block)
            
            # Update uncovered
            for ts_idx in block_tsets[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_set.remove(ts_idx)
                    
        return blocks

    def local_search(blocks_indices):
        """
        Try to remove blocks or replace them.
        blocks_indices: list of indices into all_blocks
        """
        blocks = blocks_indices[:]
        uncovered = np.zeros(num_tsets, dtype=bool)
        
        # Calculate initial coverage
        for b_idx in blocks:
            for ts_idx in block_tsets[b_idx]:
                uncovered[ts_idx] = True # True means covered? Let's use "covered" flag.
                
        # Let's use a "covered" boolean array.
        covered = np.zeros(num_tsets, dtype=bool)
        for b_idx in blocks:
            for ts_idx in block_tsets[b_idx]:
                covered[ts_idx] = True
        
        # Check if all covered
        if not np.all(covered):
            return None # Invalid
            
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # Try to remove a block
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                    
                b_idx = blocks[i]
                
                # Check if any tset is ONLY covered by this block
                # We need to know for each tset, how many blocks cover it.
                # Let's maintain a count array.
                
                # Re-calculate counts? Expensive.
                # Let's just check: for each tset covered by b_idx, is it covered by any other block in blocks?
                
                redundant = True
                for ts_idx in block_tsets[b_idx]:
                    # Check if covered by any other block
                    is_covered_by_other = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        # Check if blocks[j] covers ts_idx
                        # We can precompute a set of tsets for each block for faster lookup?
                        # Or just check if ts_idx is in block_tsets[blocks[j]]
                        # block_tsets[blocks[j]] is a numpy array.
                        # np.isin is slow.
                        # Let's convert block_tsets to sets for faster lookup?
                        # Or just assume it's fast enough for small v?
                        
                        # Optimization: Precompute block_tsets_sets
                        # I'll do this in the main scope.
                        
                        pass # Logic handled by precomputed sets below
                        
                # This loop is O(N_blocks * C(8,5) * N_blocks) which is too slow.
                # Better: Maintain a coverage count array.
                
                # Let's restart local search with coverage counts.
                return None # Fallback to simpler logic or different strategy

    # --- FINAL IMPLEMENTATION STRATEGY ---
    
    # Precompute block_tsets as sets for O(1) lookup
    block_tsets_sets = [set(block_tsets[i]) for i in range(num_blocks)]
    
    def greedy_cover_fast(seed_val):
        random.seed(seed_val)
        uncovered = np.ones(num_tsets, dtype=bool)
        uncovered_set = set(range(num_tsets))
        blocks = []
        
        while uncovered_set:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered tset
            target_tset = random.choice(list(uncovered_set))
            candidate_blocks = tset_to_blocks[target_tset]
            
            best_block = -1
            best_gain = -1
            
            for b_idx in candidate_blocks:
                gain = 0
                for ts_idx in block_tsets[b_idx]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                elif gain == best_gain and random.random() < 0.5:
                    best_block = b_idx
                    
            if best_block == -1:
                break
                
            blocks.append(best_block)
            
            for ts_idx in block_tsets[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_set.remove(ts_idx)
                    
        return blocks

    def local_search_remove(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return blocks
            
        # Maintain coverage count
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in blocks:
            for ts_idx in block_tsets[b_idx]:
                cover_count[ts_idx] += 1
                
        if np.any(cover_count == 0):
            return None # Invalid
            
        blocks_list = blocks[:]
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # Try removing each block
            for i in range(len(blocks_list)):
                if time.time() > deadline:
                    break
                    
                b_idx = blocks_list[i]
                
                # Check if safe to remove
                safe = True
                for ts_idx in block_tsets[b_idx]:
                    if cover_count[ts_idx] == 1:
                        safe = False
                        break
                        
                if safe:
                    # Remove it
                    for ts_idx in block_tsets[b_idx]:
                        cover_count[ts_idx] -= 1
                    blocks_list.pop(i)
                    improved = True
                    # Restart loop to check for new redundancies
                    break
                    
        return blocks_list

    # Run multiple attempts
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        blocks = greedy_cover_fast(attempt)
        if blocks is None:
            continue
            
        # Local search to remove redundant blocks
        blocks = local_search_remove(blocks)
        
        if blocks is None:
            # Fallback: if local search failed due to timeout or error, keep greedy result
            blocks = greedy_cover_fast(attempt)
            if blocks is None:
                continue
                
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback to a simple valid construction if all else fails (should not happen)
        # Just pick random blocks until covered?
        # For now, return empty or raise?
        # The task requires valid.
        # We will do one more greedy without local search timeout pressure?
        blocks = greedy_cover_fast(0)
        if blocks:
            return [list(all_blocks[b]) for b in blocks]
        return []
        
    return [list(all_blocks[b]) for b in best_blocks]
