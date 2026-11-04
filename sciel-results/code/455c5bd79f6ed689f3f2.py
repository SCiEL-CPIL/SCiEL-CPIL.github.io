import itertools
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Deadline: 150s limit, use 140s for safety
    deadline = time.time() + 140.0
    
    best_blocks = None
    best_len = float('inf')
    
    # Precompute the t-subsets contained in each potential block?
    # There are C(19,8) = 75582 blocks. Each has C(8,5)=56 t-subsets.
    # Total 4.2M pairs. This is manageable to precompute as a list of lists of indices.
    # However, generating all blocks and their t-subsets takes time.
    # Let's generate them on the fly or cache them if we use the same blocks often.
    # Given the number of restarts, on-the-fly might be faster if we don't reuse blocks much.
    # But in greedy, we pick specific blocks.
    
    # Helper: get indices of t-subsets in a block
    def get_block_tsub_indices(block):
        return [tsub_index[ts] for ts in itertools.combinations(block, t)]

    # Helper: Greedy construction
    def greedy_construct(seed):
        random.seed(seed)
        covered = bytearray(num_tsubs)
        uncovered_count = num_tsubs
        blocks = []
        
        while uncovered_count > 0:
            # Pick a random uncovered t-subset to anchor the block
            # This ensures we address uncovered areas
            while True:
                i = random.randint(0, num_tsubs - 1)
                if not covered[i]:
                    start_tsub = all_tsubs[i]
                    break
            
            block = list(start_tsub)
            
            # Extend to k points
            while len(block) < k:
                best_point = None
                best_gain = -1
                # Calculate gain for each candidate point
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # New t-subsets formed by adding x
                    # They are (t-1)-combinations of current block + x
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_index[ts]
                        if not covered[idx]:
                            gain += 1
                    if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    # Should not happen if v > k
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Mark as covered
            for ts in itertools.combinations(block, t):
                idx = tsub_index[ts]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1
        
        return blocks

    # Helper: Reduction phase
    # Remove blocks that are redundant
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = [0] * num_tsubs
        
        # Initialize counts
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = tsub_index[ts]
                cover_count[idx] += 1
        
        # Iteratively remove redundant blocks
        changed = True
        current_blocks = blocks[:]
        
        while changed:
            changed = False
            new_blocks = []
            for b in current_blocks:
                # Check if any t-subset in b has count == 1
                # If yes, this block is essential (cannot be removed yet)
                # If no, this block is redundant
                is_essential = False
                for ts in itertools.combinations(b, t):
                    idx = tsub_index[ts]
                    if cover_count[idx] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_blocks.append(b)
                else:
                    # Remove this block
                    changed = True
                    # Decrement counts
                    for ts in itertools.combinations(b, t):
                        idx = tsub_index[ts]
                        cover_count[idx] -= 1
            
            current_blocks = new_blocks
        
        return current_blocks

    # Main loop: Multiple restarts
    num_restarts = 200 # We will break based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # 1. Greedy Construction
        blocks = greedy_construct(restart)
        
        # 2. Reduction
        reduced_blocks = reduce_blocks(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks
        
        # If we reach a very good solution, we might want to stop or keep trying for a while
        # The best known is 335. If we get <= 330, that's great.
        if best_len <= 320:
            # Try a few more to see if we can go lower
            if restart > 10:
                break

    if best_blocks is None:
        # Fallback: One deterministic greedy + reduction
        blocks = greedy_construct(0)
        best_blocks = reduce_blocks(blocks)
        best_len = len(best_blocks)

    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
