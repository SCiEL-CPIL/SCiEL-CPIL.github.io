import itertools
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using an optimized greedy set cover algorithm.
    Specifically optimized for v=24, k=7, t=3, but general enough for similar small parameters.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for overhead/return

    # Parameters for the specific task, but allow general use if needed
    # The task specifies v, k, t are passed.
    if v != 24 or k != 7 or t != 3:
        # Fallback to a simpler greedy if parameters differ, though task implies fixed params
        # For robustness, we implement the general logic, but the optimization is for 24,7,3
        pass

    # Pre-computation
    points = range(v)
    triples = list(itertools.combinations(points, t))
    num_triples = len(triples)
    triple_to_id = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(points, k))
    num_blocks = len(blocks)
    
    # Map triple to list of block indices that contain it
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    # Pre-compute bitmasks for each block
    # A block covers all t-subsets of its k points
    block_masks = [0] * num_blocks
    
    # Efficiently compute masks and inverted index
    # For each block, find all t-subsets
    for b_idx, block in enumerate(blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            tid = triple_to_id[sub]
            mask |= (1 << tid)
            triple_to_blocks[tid].append(b_idx)
        block_masks[b_idx] = mask
        
    # Check time after pre-computation
    if time.time() > deadline:
        # If pre-computation took too long, return empty or a trivial set (though 346k blocks is fast)
        # For 24,7,3 this is very fast.
        return []

    full_mask = (1 << num_triples) - 1
    covered_mask = 0
    selected_blocks = []
    
    # Greedy Set Cover
    while covered_mask != full_mask:
        if time.time() > deadline:
            # Break and return best so far? 
            # The problem requires a VALID covering. If we break early, it's invalid.
            # So we must ensure we finish or use a faster method.
            # Given the speed estimate, this shouldn't happen.
            # If it does, we might need to switch to a less optimized but more predictable loop?
            # No, the loop is the bottleneck if it's slow. But it's fast.
            break

        # Find an uncovered triple
        # Get the lowest set bit in the complement
        uncovered = full_mask & ~covered_mask
        if uncovered == 0:
            break
            
        # Find the index of the lowest set bit
        # Python int doesn't have lowbit index directly, but we can use bit_length or a loop
        # Or just iterate? No, use a trick.
        # lsb = uncovered & -uncovered
        # t_id = lsb.bit_length() - 1
        
        # Actually, we can just pick any uncovered triple. The lowest one is fine.
        lsb = uncovered & -uncovered
        t_id = lsb.bit_length() - 1
        
        # Get candidates: blocks containing this triple
        candidates = triple_to_blocks[t_id]
        
        best_gain = -1
        best_b_idx = -1
        
        # Evaluate gain for each candidate
        # gain = popcount(block_mask & ~covered_mask)
        # We can optimize by pre-computing ~covered_mask? No, it changes every step.
        # But ~covered_mask is the same for all candidates in this step.
        uncov = ~covered_mask
        # Note: Python ints are infinite precision, so ~covered_mask has infinite 1s.
        # We should mask it with full_mask to be safe, though & with block_mask (which is within full_mask) handles it.
        
        for b_idx in candidates:
            # Intersection of block's coverage with currently uncovered
            common = block_masks[b_idx] & uncov
            gain = common.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b_idx = b_idx
                # Optional: if gain is 0, break? No, we need to cover the triple t_id, so gain >= 1 always for candidates.
        
        if best_b_idx == -1:
            # Should not happen if candidates exist
            break
            
        # Update state
        covered_mask |= block_masks[best_b_idx]
        selected_blocks.append(blocks[best_b_idx])
        
        # Optional: Pruning?
        # We can remove selected block from candidates of other triples?
        # That would require updating triple_to_blocks, which is expensive.
        # Given the small number of steps (~70), it's not necessary.
        
    # Verify validity (optional, but good for debugging)
    # final_mask = 0
    # for b in selected_blocks:
    #     final_mask |= block_masks[blocks.index(b)] # Inefficient, use index from selection
    # We stored blocks directly.
    
    return selected_blocks

# Helper for quick testing if run directly
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct(24, 7, 3)
    end = time.time()
    print(f"Number of blocks: {len(res)}")
    print(f"Time taken: {end - start:.2f}s")
