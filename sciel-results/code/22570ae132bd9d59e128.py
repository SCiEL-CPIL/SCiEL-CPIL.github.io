import itertools
import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

# Global Precomputation
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
    deadline = start_time + 118  # Safety margin

    def greedy_init(seed):
        """Greedy set cover with randomization for tie-breaking."""
        rng = random.Random(seed)
        counts = [0] * N_SUBSETS
        current_blocks = []
        current_masks = []
        
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

        uncovered_mask = (1 << N_SUBSETS) - 1
        while uncovered_mask:
            best_idx = -1
            best_gain = 0
            # Use randomization to break ties and explore different solutions
            for i in range(N_BLOCKS):
                gain = (all_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0 and rng.random() < 0.1:
                    best_idx = i
            if best_idx == -1:
                break
            add_block(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
        
        return current_blocks, current_masks, counts

    def is_redundant(i, counts):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    def local_search(current_blocks, current_masks, counts, seed):
        """ILS-style local search with pruning and swaps."""
        rng = random.Random(seed)
        best_len = len(current_blocks)
        best_blocks = current_blocks.copy()
        best_masks = current_masks.copy()
        best_counts = counts.copy()
        
        no_improve_count = 0
        max_no_improve = 30
        max_no_improve_base = 30
        
        while time.time() < deadline:
            # Phase 1: Prune redundant blocks
            removed_any = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                if is_redundant(i, counts):
                    # Remove block i
                    m = current_masks[i]
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        mm ^= lsb
                    current_blocks.pop(i)
                    current_masks.pop(i)
                    removed_any = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                        best_masks = current_masks.copy()
                        best_counts = counts.copy()
                    break
            
            if removed_any:
                no_improve_count = 0
                continue
            
            # Phase 2: Swap
            if len(current_blocks) == 0:
                break
            
            # Pick a random block to remove
            i = rng.randint(0, len(current_blocks) - 1)
            
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
                m = current_masks[i]
                mm = m
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    mm ^= lsb
                current_blocks.pop(i)
                current_masks.pop(i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                    best_masks = current_masks.copy()
                    best_counts = counts.copy()
                no_improve_count = 0
                continue
            
            # If too many bits to cover, this swap is likely hard.
            if len(uncovered_bits) > 3:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation: Add a block that covers many well-covered areas
                    # to enable new pruning opportunities
                    # Score blocks by how many bits with count >= 2 they cover
                    best_perturb_idx = -1
                    best_perturb_score = -1
                    for c in range(N_BLOCKS):
                        if c in current_blocks:
                            continue
                        score = 0
                        m_c = all_masks[c]
                        mm_c = m_c
                        while mm_c:
                            lsb = mm_c & -mm_c
                            bit_idx = lsb.bit_length() - 1
                            if counts[bit_idx] >= 2:
                                score += 1
                            mm_c ^= lsb
                        if score > best_perturb_score:
                            best_perturb_score = score
                            best_perturb_idx = c
                    if best_perturb_idx >= 0:
                        # Add the perturbation block
                        current_blocks.append(best_perturb_idx)
                        current_masks.append(all_masks[best_perturb_idx])
                        m = all_masks[best_perturb_idx]
                        mm = m
                        while mm:
                            lsb = mm & -mm
                            bit_idx = lsb.bit_length() - 1
                            counts[bit_idx] += 1
                            mm ^= lsb
                        no_improve_count = 0
                        max_no_improve = min(100, max_no_improve_base * 2)
                    else:
                        # Fallback: random perturbation
                        c = rng.randint(0, N_BLOCKS - 1)
                        if c not in current_blocks:
                            current_blocks.append(c)
                            current_masks.append(all_masks[c])
                            m = all_masks[c]
                            mm = m
                            while mm:
                                lsb = mm & -mm
                                bit_idx = lsb.bit_length() - 1
                                counts[bit_idx] += 1
                                mm ^= lsb
                            no_improve_count = 0
                            max_no_improve = min(100, max_no_improve_base * 2)
                    continue
            else:
                # Find a block that covers all uncovered_bits
                # Sort by frequency to speed up intersection
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                # Start with the smallest list
                candidates = set(blocks_containing_subset[uncovered_bits[0]])
                
                # Intersect with others
                for b in uncovered_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                
                if candidates:
                    # Exclude the block we're removing
                    candidates.discard(i)
                    if candidates:
                        j = rng.choice(list(candidates))
                        # Perform swap: remove i, add j
                        m_i = current_masks[i]
                        mm = m_i
                        while mm:
                            lsb = mm & -mm
                            bit_idx = lsb.bit_length() - 1
                            counts[bit_idx] -= 1
                            mm ^= lsb
                        current_blocks.pop(i)
                        current_masks.pop(i)
                        
                        current_blocks.append(j)
                        current_masks.append(all_masks[j])
                        m_j = all_masks[j]
                        mm = m_j
                        while mm:
                            lsb = mm & -mm
                            bit_idx = lsb.bit_length() - 1
                            counts[bit_idx] += 1
                            mm ^= lsb
                        no_improve_count = 0
                        continue
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Perturbation
                        c = rng.randint(0, N_BLOCKS - 1)
                        if c not in current_blocks:
                            current_blocks.append(c)
                            current_masks.append(all_masks[c])
                            m = all_masks[c]
                            mm = m
                            while mm:
                                lsb = mm & -mm
                                bit_idx = lsb.bit_length() - 1
                                counts[bit_idx] += 1
                                mm ^= lsb
                            no_improve_count = 0
                            max_no_improve = min(100, max_no_improve_base * 2)
                        continue
        
        return best_blocks, best_masks, best_counts

    # Run multiple independent searches with different seeds
    best_overall_blocks = None
    best_overall_len = float('inf')
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + restart * 137
        current_blocks, current_masks, counts = greedy_init(seed)
        
        # Run local search
        best_blocks, best_masks, best_counts = local_search(current_blocks, current_masks, counts, seed)
        
        if len(best_blocks) < best_overall_len:
            best_overall_len = len(best_blocks)
            best_overall_blocks = best_blocks
    
    if best_overall_blocks is None:
        # Fallback: return greedy result
        current_blocks, current_masks, counts = greedy_init(42)
        best_overall_blocks = current_blocks
    
    return [list(all_blocks[i]) for i in best_overall_blocks]
