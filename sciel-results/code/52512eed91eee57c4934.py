import itertools
import random
import time

# Global Precomputation for v=19, k=7, t=4
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
# For each subset, list of blocks containing it
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
# For each block, list of subset indices it covers
all_block_subsets = []

for block in itertools.combinations(range(V), K):
    mask = 0
    subs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        subs.append(idx)
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)
    all_block_subsets.append(subs)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_len = float('inf')
    best_blocks = []
    
    # Use a fixed number of restarts, but adapt based on time
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Set random seed for reproducibility within restart but diversity across restarts
        random.seed(12345 + restart)
        
        current_blocks = []
        current_masks = []
        counts = [0] * N_SUBSETS
        in_set = set()
        
        def add_block(idx):
            current_blocks.append(idx)
            current_masks.append(all_masks[idx])
            in_set.add(idx)
            for bit_idx in all_block_subsets[idx]:
                counts[bit_idx] += 1

        def remove_block(idx):
            pos = current_blocks.index(idx)
            m = current_masks[pos]
            for bit_idx in all_block_subsets[idx]:
                counts[bit_idx] -= 1
            current_blocks.pop(pos)
            current_masks.pop(pos)
            in_set.discard(idx)

        def is_redundant(idx):
            for bit_idx in all_block_subsets[idx]:
                if counts[bit_idx] == 1:
                    return False
            return True

        # 1. Fast Greedy Initialization with Randomized Tie-Breaking
        uncovered_mask = (1 << N_SUBSETS) - 1
        while uncovered_mask:
            best_gain = 0
            candidates = []
            for i in range(N_BLOCKS):
                if i in in_set:
                    continue
                gain = (all_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            best_idx = random.choice(candidates)
            add_block(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
            
        # Initial pruning - remove redundant blocks
        i = 0
        while i < len(current_blocks):
            if is_redundant(current_blocks[i]):
                remove_block(current_blocks[i])
            else:
                i += 1
                
        current_len = len(current_blocks)
        if current_len < best_len:
            best_len = current_len
            best_blocks = current_blocks.copy()
            
        # 2. Iterative Local Search
        no_improve_count = 0
        max_no_improve = 50
        
        # Allocate remaining time among restarts
        remaining_time = deadline - start_time
        restarts_done = restart + 1
        time_per_restart = remaining_time / (num_restarts - restarts_done + 1)
        restart_deadline = start_time + time_per_restart
        
        while time.time() < restart_deadline:
            # Phase 1: Prune - try to remove any redundant block
            removed_any = False
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > restart_deadline:
                    break
                idx = current_blocks[i]
                if is_redundant(idx):
                    remove_block(idx)
                    removed_any = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    break
            
            if removed_any:
                no_improve_count = 0
                continue
            
            # Phase 2: Try 1-out/1-in swap
            if len(current_blocks) == 0:
                break
                
            # Pick a random block to remove
            i = random.randint(0, len(current_blocks) - 1)
            idx_i = current_blocks[i]
            
            # Find subsets that would become uncovered
            uncovered_bits = []
            for bit_idx in all_block_subsets[idx_i]:
                if counts[bit_idx] == 1:
                    uncovered_bits.append(bit_idx)
                    
            if not uncovered_bits:
                # It was redundant, remove it
                remove_block(idx_i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                continue
                
            # If too many bits to cover, this swap is likely hard.
            if len(uncovered_bits) > 4:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation: Add a random block, then prune
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in in_set:
                        add_block(c)
                        no_improve_count = 0
                        max_no_improve = min(200, max_no_improve * 2)
                    continue
            else:
                # Efficient Swap Search:
                # 1. Create a bitmask of the critical subsets
                critical_mask = 0
                for b in uncovered_bits:
                    critical_mask |= 1 << b
                    
                # 2. Find the most constrained subset (fewest covering blocks)
                min_len = float('inf')
                best_subset = uncovered_bits[0]
                for b in uncovered_bits:
                    l = len(blocks_containing_subset[b])
                    if l < min_len:
                        min_len = l
                        best_subset = b
                        
                # 3. Iterate through blocks containing the most constrained subset
                candidates = blocks_containing_subset[best_subset]
                valid_candidates = []
                
                for j in candidates:
                    if j == idx_i or j in in_set:
                        continue
                    # Check if j covers ALL critical subsets
                    if (all_masks[j] & critical_mask) == critical_mask:
                        valid_candidates.append(j)
                        
                if valid_candidates:
                    # Pick a random valid candidate
                    j = random.choice(valid_candidates)
                    # Perform swap
                    remove_block(idx_i)
                    add_block(j)
                    no_improve_count = 0
                    continue
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Perturbation: try 2-in/1-out or add random
                        if random.random() < 0.5 and len(current_blocks) > 0:
                            # Try to add a block that helps cover critical subsets
                            for _ in range(10):
                                c = random.randint(0, N_BLOCKS - 1)
                                if c not in in_set:
                                    add_block(c)
                                    no_improve_count = 0
                                    max_no_improve = min(200, max_no_improve * 2)
                                    break
                        else:
                            c = random.randint(0, N_BLOCKS - 1)
                            if c not in in_set:
                                add_block(c)
                                no_improve_count = 0
                                max_no_improve = min(200, max_no_improve * 2)
                        continue

    # Final cleanup: ensure we return the best found
    return [list(all_blocks[i]) for i in best_blocks]
