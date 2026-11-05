import itertools
import random
import time
import numpy as np

# Global Precomputation for v=19, k=7, t=4
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets) # 3876

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
# For each subset, list of blocks containing it
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
    deadline = start_time + 115  # Safety margin
    
    best_len = float('inf')
    best_blocks = []
    
    # Simulated Annealing with multiple restarts
    num_restarts = 25
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        random.seed(2000 + restart)
        
        current_blocks = []
        counts = [0] * N_SUBSETS
        block_in_use = [False] * N_BLOCKS
        
        def add_block(idx):
            current_blocks.append(idx)
            block_in_use[idx] = True
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                mm ^= lsb

        def remove_block(i):
            idx = current_blocks[i]
            block_in_use[idx] = False
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] -= 1
                mm ^= lsb
            current_blocks.pop(i)

        def is_redundant(i):
            m = all_masks[current_blocks[i]]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    return False
                mm ^= lsb
            return True

        # 1. Greedy Initialization
        uncovered_mask = (1 << N_SUBSETS) - 1
        while uncovered_mask:
            best_gain = 0
            candidates = []
            for i in range(N_BLOCKS):
                if not block_in_use[i]:
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
        
        # Initial Pruning
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
            
        # 2. Simulated Annealing
        # Temperature schedule: exponential cooling
        T_init = 10.0
        T_final = 0.1
        cooling_rate = (T_final / T_init) ** (1.0 / max(1, int(time_per_restart / 0.01)))
        
        current_temp = T_init
        restart_deadline = start_time + time_per_restart * (restart + 1)
        
        while time.time() < restart_deadline:
            # Try random move: remove one block, add another
            if len(current_blocks) == 0:
                break
                
            i = random.randint(0, len(current_blocks) - 1)
            idx_i = current_blocks[i]
            m_i = all_masks[idx_i]
            
            # Find subsets that would become uncovered if we remove i
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
                continue
            
            # If we have lost subsets, try to find a block j that covers them
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            
            # Start with the rarest subset's blocks
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            # Intersect with other subsets' blocks
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            # Filter out blocks already in use
            valid_candidates = [c for c in candidates if c != idx_i and not block_in_use[c]]
            
            if valid_candidates:
                j = random.choice(valid_candidates)
                
                # Calculate delta in objective (number of blocks)
                # We're doing a swap: remove i, add j
                # Objective is len(current_blocks), so delta = 0 for swap
                # But we need to check if this maintains coverage
                # For SA, we accept based on temperature
                
                # Try the swap
                remove_block(i)
                add_block(j)
                
                # Check if coverage is still valid (all counts >= 1)
                # Actually, we need to verify that the new block covers all uncovered_bits
                # and doesn't create issues
                # Since we selected j from intersection of blocks containing all uncovered_bits,
                # coverage should be maintained
                
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
            else:
                # No valid swap found, try perturbation: add a random block
                c = random.randint(0, N_BLOCKS - 1)
                if not block_in_use[c]:
                    add_block(c)
                    # Then try to prune
                    i = 0
                    while i < len(current_blocks):
                        if is_redundant(i):
                            remove_block(i)
                        else:
                            i += 1
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
            
            # Cool down
            current_temp *= cooling_rate
            if current_temp < T_final:
                current_temp = T_final

    # Final cleanup
    return [list(all_blocks[i]) for i in best_blocks]
