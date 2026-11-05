import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Precompute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute bitsets for all possible blocks
    # This is the expensive part, but it's a one-time cost.
    # We store:
    # 1. all_blocks: list of block tuples
    # 2. all_block_bitsets: list of ints representing covered triples
    
    all_blocks = []
    all_block_bitsets = []
    
    # Optimization: Use a list for bitsets to allow fast access.
    # We don't need a dict mapping block->index if we just iterate by index.
    
    try:
        for b in itertools.combinations(range(v), k):
            bitset = 0
            for tri in itertools.combinations(b, t):
                idx = triple_to_idx[tri]
                bitset |= (1 << idx)
            all_blocks.append(b)
            all_block_bitsets.append(bitset)
    except Exception as e:
        # Fallback if precomputation fails
        return []

    num_blocks = len(all_blocks)
    
    # --- Phase 1: Candidate-Pool Greedy Construction ---
    # Instead of scanning all blocks every step, we maintain a pool of candidates.
    # Initially, the pool contains a random sample or all blocks (if N is small enough).
    # Here N=346k, so we can't keep all in a "top-k" heap easily without overhead,
    # but we can keep a list of "active" candidates.
    
    # Strategy:
    # 1. Start with a random subset of blocks in the pool? No, that might miss good ones.
    # 2. Better: Start with ALL blocks in a list.
    # 3. At each step, find the best in the pool.
    # 4. After selection, update gains for the remaining pool?
    #    Updating gains for 346k blocks is O(N) per step. 76 * 346k = 26M ops.
    #    This is much faster than 76 * 346k * (popcount cost) if popcount is expensive.
    #    Actually, popcount of a 2024-bit int is fast in Python (C implementation).
    #    The bottleneck in the original was likely the Python loop overhead of iterating 346k items 76 times.
    #    
    # Let's try to speed up the greedy selection.
    # We can use NumPy?
    # Convert bitsets to a 2D array? 346k * 2024 bits is too big for a single array of ints?
    # 346k * 2024 / 8 bytes ~ 87 MB. That's feasible.
    # But bitwise AND and popcount in NumPy might be slower per element than Python int for small counts?
    # Python int bit ops are C-speed.
    
    # Let's stick to Python ints but optimize the loop.
    
    uncovered_bitset = (1 << num_triples) - 1
    selected_indices = []
    active_indices = list(range(num_blocks))
    
    # To speed up, we can pre-filter blocks that are "bad" initially?
    # All blocks cover C(7,3)=35 triples.
    
    # Greedy Loop
    while uncovered_bitset != 0:
        best_gain = -1
        best_idx = -1
        
        # Iterate through active candidates
        # Optimization: If the pool gets large, we can sample?
        # No, for correctness of greedy, we need the best.
        # But we can break early if gain is 0? No, we need max gain.
        
        # Let's use a local variable for speed
        unc = uncovered_bitset
        best_g = -1
        best_i = -1
        
        for i in active_indices:
            bset = all_block_bitsets[i]
            # Fast path: if bset is 0, skip (not possible here)
            # Gain calculation
            g = (bset & unc).bit_count()
            
            if g > best_g:
                best_g = g
                best_i = i
                
        if best_i == -1:
            break
            
        selected_indices.append(best_i)
        
        # Remove from active
        # Removing from a list is O(N).
        # We can mark as inactive or swap-remove.
        # Swap-remove is O(1) average if we know the index in active_indices.
        # But we don't have a map from block_idx to active_idx.
        # We can just filter active_indices at the end? No, that's O(N) per step anyway.
        # Let's just use a set of selected indices and skip them?
        # The original did that.
        
        # Let's try a different approach:
        # Keep a list of candidates.
        # After each step, we can re-evaluate only the candidates that might have changed?
        # Gains only decrease.
        
        # Actually, the biggest speedup comes from using a heap?
        # We can push (gain, idx) into a heap.
        # But gains change.
        
        # Let's just run the standard greedy but with a tight time check.
        # If it's too slow, we switch to a heuristic.
        
        # Update uncovered
        uncovered_bitset &= ~all_block_bitsets[best_i]
        
        # Remove best_i from active_indices
        # To avoid O(N) removal, we can just leave it and check if it's selected?
        # But we already did that in the original and it was 1.09.
        
        # Let's try to optimize the inner loop by using a precomputed list of gains?
        # No, gains depend on uncovered.
        
        # Let's try to use a "lazy" deletion heap.
        # Heap of (-gain, idx).
        # When we pop, if the gain is stale (recomputed gain != stored gain), push the new one.
        # This requires O(log N) per update.
        # But we have N items.
        
        # Let's just try to make the current loop as fast as possible.
        # Maybe pre-compute the list of bitsets as a local variable.
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    # If we broke out early, we might not have a full cover.
    # Let's check coverage.
    # If not covered, we need to add more blocks.
    # For simplicity, if we run out of time, we return what we have (which might be invalid).
    # The task says "Valid when every t-subset... lies in at least one block".
    # If we return an invalid cover, score is 0.
    # So we MUST ensure validity.
    
    # Let's check if uncovered_bitset is 0.
    if uncovered_bitset != 0:
        # We need to cover the rest.
        # Use a fast heuristic: pick random blocks containing uncovered triples.
        # This might not be optimal but ensures validity.
        
        # Find uncovered triples
        # We need to map bitset to triples.
        # This is slow.
        
        # Alternative: Just continue greedy but with a smaller pool?
        # Or just add random blocks until covered.
        
        # Let's try to complete with random blocks.
        # This is a fallback.
        
        # Identify uncovered triples from bitset
        # This is O(num_triples)
        unc_triples = []
        for i in range(num_triples):
            if (uncovered_bitset >> i) & 1:
                unc_triples.append(all_triples[i])
                
        # For each uncovered triple, pick a random block containing it?
        # That might add too many blocks.
        
        # Better: Pick a random block from the remaining active ones?
        # Or just pick random blocks until covered.
        
        # Let's pick random blocks from the full set.
        while uncovered_bitset != 0:
            # Pick a random block
            idx = random.randint(0, num_blocks - 1)
            bset = all_block_bitsets[idx]
            if bset & uncovered_bitset:
                selected_indices.append(idx)
                uncovered_bitset &= ~bset
            
            if time.time() > deadline - 5:
                # If we run out of time, we have to stop.
                # But we must return a valid cover.
                # If we can't, we return empty? No, score 0.
                # Let's hope we finish.
                break
                
    # Convert to list of blocks
    selected_blocks = [list(all_blocks[i]) for i in selected_indices]
    
    # --- Phase 2: Optimization (Redundancy Removal & Local Search) ---
    
    # Redundancy Removal
    def optimize_redundancy(blocks):
        if not blocks:
            return []
        
        # Build counts
        counts = {}
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[tri] = counts.get(tri, 0) + 1
                
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if not active[i]:
                    continue
                b = blocks[i]
                can_remove = True
                for tri in itertools.combinations(b, t):
                    if counts[tri] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active[i] = False
                    changed = True
                    for tri in itertools.combinations(b, t):
                        counts[tri] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active[i]]
        
    final_blocks = optimize_redundancy(selected_blocks)
    
    # Local Search: Try to swap blocks to reduce count?
    # Since we already removed redundancy, the count is minimal for this set.
    # To reduce count further, we need to replace a set of blocks with fewer blocks.
    # This is hard.
    
    # Let's try a simple 1-swap: Replace block A with block B if B covers all triples of A
    # that are not covered by others?
    # That's just redundancy removal again.
    
    # What if we replace 2 blocks with 1?
    # Check if union of triples of A and B is covered by a single block C?
    # If so, replace A, B with C.
    
    # This is a 2-for-1 replacement.
    # Let's implement this if we have time.
    
    if time.time() < deadline - 20:
        # 2-for-1 replacement
        # Precompute for each block the set of triples it covers (as a set of indices or tuple)
        block_triples = []
        for b in final_blocks:
            tris = tuple(sorted(itertools.combinations(b, t)))
            block_triples.append(tris)
            
        # Map triple -> list of block indices covering it
        tri_to_blocks = {}
        for i, tris in enumerate(block_triples):
            for tri in tris:
                if tri not in tri_to_blocks:
                    tri_to_blocks[tri] = []
                tri_to_blocks[tri].append(i)
                
        # Try to find pairs (i, j) such that union(tris_i, tris_j) is subset of tris_k for some k
        # This is expensive.
        
        # Alternative: For each block k, check if there are two blocks i, j such that
        # tris_i | tris_j <= tris_k.
        # If so, remove i, j and add k (if not already present).
        
        # This is complex. Let's stick to redundancy removal for now.
        
    return final_blocks
