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
# For each subset, list of blocks containing it (for efficient swap search)
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
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
    
    # We will run multiple ILS instances with different random seeds
    # to explore different regions of the solution space.
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        # Set random seed for reproducibility within restart but diversity across restarts
        random.seed(12345 + restart)
        
        current_blocks = []
        current_masks = []
        counts = [0] * N_SUBSETS
        
        def add_block(idx):
            current_blocks.append(idx)
            current_masks.append(all_masks[idx])
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                mm ^= lsb

        def remove_block(i):
            m = current_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] -= 1
                mm ^= lsb
            current_blocks.pop(i)
            current_masks.pop(i)

        def is_redundant(i):
            m = current_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    return False
                mm ^= lsb
            return True

        # 1. Greedy Initialization with Randomized Tie-Breaking
        uncovered_mask = (1 << N_SUBSETS) - 1
        while uncovered_mask:
            best_gain = 0
            candidates = []
            for i in range(N_BLOCKS):
                gain = (all_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Randomly pick from best candidates to diversify starts
            best_idx = random.choice(candidates)
            add_block(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
            
        # Initial pruning
        i = 0
        while i < len(current_blocks):
            if is_redundant(i):
                remove_block(i)
            else:
                i += 1
                
        current_len = len(current_blocks)
        if current_len < best_len:
            best_len = current_len
            best_blocks = current_blocks.copy()
            
        # 2. Iterative Local Search
        # We alternate between:
        # a. Pruning redundant blocks
        # b. Swapping blocks to enable further pruning
        # c. Perturbing if stuck
        
        no_improve_count = 0
        max_no_improve = 50
        
        restart_deadline = start_time + time_per_restart * (restart + 1)
        
        while time.time() < restart_deadline:
            # Phase 1: Prune
            removed_any = False
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > restart_deadline:
                    break
                if is_redundant(i):
                    remove_block(i)
                    removed_any = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    # Restarting the loop is safer after removal
                    break
            
            if removed_any:
                no_improve_count = 0
                continue
            
            # Phase 2: Swap
            if len(current_blocks) == 0:
                break
                
            # Pick a random block to remove
            i = random.randint(0, len(current_blocks) - 1)
            
            # Find subsets that would become uncovered
            m_i = current_masks[i]
            uncovered_bits = []
            mm = m_i
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    uncovered_bits.append(bit_idx)
                mm ^= lsb
                
            if not uncovered_bits:
                # It was redundant, remove it
                remove_block(i)
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
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                        max_no_improve = min(200, max_no_improve * 2)
                    continue
            else:
                # Find a block that covers all uncovered_bits
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                # Start with the smallest list
                candidates = blocks_containing_subset[uncovered_bits[0]]
                
                # Intersect with others
                for b in uncovered_bits[1:]:
                    if not candidates:
                        break
                    set_cands = set(candidates)
                    set_b = set(blocks_containing_subset[b])
                    candidates = list(set_cands.intersection(set_b))
                    
                if candidates:
                    # Exclude the block we are removing
                    valid_candidates = [c for c in candidates if c != i]
                    if valid_candidates:
                        j = random.choice(valid_candidates)
                        # Perform swap
                        remove_block(i)
                        add_block(j)
                        no_improve_count = 0
                        continue
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Perturbation
                        c = random.randint(0, N_BLOCKS - 1)
                        if c not in current_blocks:
                            add_block(c)
                            no_improve_count = 0
                            max_no_improve = min(200, max_no_improve * 2)
                        continue

    # Final cleanup: ensure we return the best found
    return [list(all_blocks[i]) for i in best_blocks]
