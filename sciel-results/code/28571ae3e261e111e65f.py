"""Greedy covering with fast gain computation, iterative pruning, and deadline guard."""
import itertools
import time

def construct(v, k, t):
    deadline = time.monotonic() + 115.0  # safety margin under 120s
    points = list(range(v))
    
    # Generate all t-subsets and assign them indices for fast lookup
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Boolean array to track covered t-subsets
    covered = [False] * num_tsubs
    blocks = []
    
    # Precompute all possible k-blocks to avoid regenerating in loop?
    # C(19, 7) is ~50k. Generating them once is cheap.
    all_kblocks = list(itertools.combinations(points, k))
    
    # For each k-block, precompute the list of t-subset indices it covers.
    # This allows O(1) or O(C(k,t)) gain calculation instead of O(C(k,t) * lookup).
    block_tsubs = []
    for b in all_kblocks:
        b_tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(b_tsubs)
    
    while True:
        if time.monotonic() > deadline:
            break
            
        best_gain = 0
        best_block_idx = -1
        
        # Find the block with maximum gain
        for i, b_tsubs in enumerate(block_tsubs):
            gain = 0
            for idx in b_tsubs:
                if not covered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        
        if best_gain == 0:
            # No more blocks can cover anything, or all are covered
            break
            
        # Select the best block
        best_block = all_kblocks[best_block_idx]
        blocks.append(list(best_block))
        
        # Mark its t-subsets as covered
        for idx in block_tsubs[best_block_idx]:
            covered[idx] = True
            
        # Check if all covered
        if all(covered):
            break
            
    return blocks
