"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

# Pre-computation for v=19, k=8, t=5
V = 19
K = 8
T = 5

# Generate all t-subsets and map them to indices
all_t_subsets = list(itertools.combinations(range(V), T))
t_subset_index = {subset: i for i, subset in enumerate(all_t_subsets)}
num_t_subsets = len(all_t_subsets)

# Generate all k-blocks and their bitmasks
all_k_blocks = list(itertools.combinations(range(V), K))
block_bitmasks = []
block_indices = [] # index in all_k_blocks

for block in all_k_blocks:
    mask = 0
    for t_sub in itertools.combinations(block, T):
        idx = t_subset_index[t_sub]
        mask |= (1 << idx)
    block_bitmasks.append(mask)

# Function to run greedy with a specific seed for tie-breaking
def run_greedy(seed):
    rng = random.Random(seed)
    uncovered = (1 << num_t_subsets) - 1
    blocks = []
    used_block_indices = set()
    
    while uncovered:
        best_gain = -1
        best_block_idx = -1
        # Iterate through all blocks to find the one covering the most uncovered subsets
        # Optimization: We can break early if we find a block covering all remaining? Unlikely.
        # We can also skip blocks that are "too similar" to existing ones? 
        # For now, scan all.
        
        # To add randomness, we can add a small random noise to the gain
        # or just randomize the order of blocks if gains are equal.
        
        # Efficient scan
        for i, mask in enumerate(block_bitmasks):
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
            elif gain == best_gain and best_block_idx != -1:
                # Tie-breaking: random selection among best
                if rng.random() < 0.5:
                    best_block_idx = i

        if best_gain == 0:
            # No block covers any new subset? Should not happen if uncovered is not 0
            # and we have all blocks. But if we restrict blocks, it might.
            # Here we have all blocks, so this shouldn't happen unless uncovered is 0.
            break
        
        # Add the block
        blocks.append(all_k_blocks[best_block_idx])
        used_block_indices.add(best_block_idx)
        uncovered &= ~block_bitmasks[best_block_idx]
        
    # Polish: Remove redundant blocks
    # Iterate through blocks and try to remove them
    # We need to check if removing a block leaves any 5-subset uncovered.
    # We can maintain a count of how many times each 5-subset is covered.
    
    # Re-calculate coverage counts for the final set of blocks
    cover_counts = [0] * num_t_subsets
    for block in blocks:
        mask = 0
        for t_sub in itertools.combinations(block, T):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        # Add mask to counts? No, we need counts per subset.
        # It's faster to use the bitmask to update counts if we had an array of ints.
        # But bit_count on (mask & uncovered) was for greedy.
        # For polish, let's just re-verify validity by checking if the union of masks is full.
        
    # Simpler Polish:
    # Keep a set of covered subsets for the current list of blocks.
    # If we remove a block, we need to know if it uniquely covered any subset.
    
    # Let's use a counter array.
    counts = [0] * num_t_subsets
    for block in blocks:
        for t_sub in itertools.combinations(block, T):
            counts[t_subset_index[t_sub]] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            # Check if any subset in this block has count 1
            unique = False
            for t_sub in itertools.combinations(block, T):
                idx = t_subset_index[t_sub]
                if counts[idx] == 1:
                    unique = True
                    break
            if not unique:
                # Remove block
                for t_sub in itertools.combinations(block, T):
                    idx = t_subset_index[t_sub]
                    counts[idx] -= 1
                blocks.pop(i)
                changed = True
                break # Restart loop as indices changed
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # If parameters differ from pre-computed, we would need to recompute.
    # The task specifies v=19, k=8, t=5.
    if v != 19 or k != 8 or t != 5:
        # Fallback to simple greedy for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    best_blocks = []
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        blocks = run_greedy(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        seed += 1
        
        # If we hit a very low number, we might stop early?
        # Best known is 335. If we get < 335, that's a new record (unlikely in 120s but possible).
        if best_count <= 335:
            # We can try to polish more, but let's just break if we are very fast
            pass

    return best_blocks

# EVOLVE-BLOCK-END
