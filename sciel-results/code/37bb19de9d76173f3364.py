import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115  # Safety margin for 120s limit
    
    rng = random.Random(1234567890)

    # Precompute all k-subsets (blocks) and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}

    # Precompute block masks as bitsets (Python ints) for fast bitwise ops
    block_masks = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            mask |= 1 << si
        block_masks[bi] = mask

    # Helper: compute coverage counts from a list of block indices
    def get_counts(blocks_list):
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in blocks_list:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        return counts

    # Helper: prune redundant blocks
    def prune(blocks_list):
        chosen = list(blocks_list)
        if not chosen:
            return chosen
        counts = get_counts(chosen)
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen)):
                bi = chosen[i]
                mask = block_masks[bi]
                redundant = True
                m = mask
                while m:
                    low = m & -m
                    si = low.bit_length() - 1
                    if counts[si] == 1:
                        redundant = False
                        break
                    m ^= low
                if redundant:
                    m = mask
                    while m:
                        low = m & -m
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        m ^= low
                    chosen.pop(i)
                    changed = True
                    break
        return chosen

    # Helper: Greedy Set Cover initialization
    def greedy_init():
        uncovered = (1 << n_subsets) - 1
        chosen = []
        chosen_set = set()
        while uncovered:
            best_bi = -1
            best_gain = 0
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if best_gain == uncovered.bit_count():
                        break
            if best_bi < 0:
                break
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            uncovered &= ~block_masks[best_bi]
        return chosen

    init_blocks = greedy_init()
    pruned_init = prune(init_blocks)
    
    global_best = list(pruned_init)
    global_best_size = len(pruned_init)
    
    # ILS with Error Repair
    # We maintain a single current solution but use a "repair" step after perturbation.
    # To allow rejection, we keep a copy of the solution before perturbation.
    
    current = list(pruned_init)
    counts = get_counts(current)
    
    temp = 5.0
    cooling = 0.999
    all_block_indices = list(range(n_blocks))
    
    while time.time() < deadline - 2:
        # Save state for potential rejection
        prev_solution = list(current)
        prev_counts = counts.copy()
        prev_size = len(current)
        
        # Perturbation: Swap 1-2 random blocks
        n_swaps = rng.randint(1, 2)
        for _ in range(n_swaps):
            if not current:
                break
            idx_out = rng.randrange(len(current))
            bi_out = current[idx_out]
            bi_in = rng.choice(all_block_indices)
            
            # Ensure bi_in is not already in current
            if bi_in in current:
                candidates = [b for b in all_block_indices if b not in current]
                if not candidates:
                    continue
                bi_in = rng.choice(candidates)
            
            # Apply swap
            mask_out = block_masks[bi_out]
            m = mask_out
            while m:
                low = m & -m
                si = low.bit_length() - 1
                counts[si] -= 1
                m ^= low
                
            mask_in = block_masks[bi_in]
            m = mask_in
            while m:
                low = m & -m
                si = low.bit_length() - 1
                counts[si] += 1
                m ^= low
                
            current[idx_out] = bi_in
        
        # Error Repair: Ensure validity
        zero_indices = np.where(counts == 0)[0]
        if len(zero_indices) > 0:
            unc_bits = 0
            for idx in zero_indices:
                unc_bits |= 1 << idx
            
            # Greedily add blocks to cover unc_bits
            max_repairs = 5
            for _ in range(max_repairs):
                if unc_bits == 0:
                    break
                best_bi = -1
                best_gain = 0
                # Optimization: Limit search space for repair? 
                # Full search is O(N). With 50k blocks, it's ~50k ops.
                # 5 repairs * 50k = 250k ops. 
                # If we do this 1000 times, 250M ops. Might be too slow in Python.
                # Let's try to optimize by checking a random subset first?
                # Or, precompute for each t-subset, a list of blocks covering it?
                # That's expensive memory-wise but fast lookup.
                # Let's stick to full search but break early if gain is high.
                
                for bi in all_block_indices:
                    if bi in current:
                        continue
                    gain = (block_masks[bi] & unc_bits).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                        if best_gain == unc_bits.bit_count():
                            break
                
                if best_bi < 0 or best_gain == 0:
                    break
                
                current.append(best_bi)
                mask_in = block_masks[best_bi]
                m = mask_in
                while m:
                    low = m & -m
                    si = low.bit_length() - 1
                    counts[si] += 1
                    m ^= low
                unc_bits &= ~mask_in
        
        # Acceptance
        new_size = len(current)
        delta = new_size - prev_size
        
        # We only accept if valid (unc_bits == 0) and (delta < 0 or prob)
        # Since we repaired, it should be valid.
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            # Accept
            if new_size < global_best_size:
                global_best_size = new_size
                global_best = list(current)
        else:
            # Reject: Revert
            current = prev_solution
            counts = prev_counts
        
        temp *= cooling

    # Final Prune
    final_solution = prune(global_best)
    
    # Verify
    counts_final = get_counts(final_solution)
    if any(c == 0 for c in counts_final):
        final_solution = pruned_init
    
    return [list(all_blocks[bi]) for bi in final_solution]
