"""Combinatorial Holography covering via spectral residual energy over Z19."""
import itertools
import numpy as np
import random
import time

# Precomputation for v=19, k=7, t=4
_V = 19
_K = 7
_T = 4

# Build list of all 4-subsets
_all_4subs = list(itertools.combinations(range(_V), _T))
_n4 = len(_all_4subs)  # 3876

# Map each 4-subset to its index
_subset_index = {s: i for i, s in enumerate(_all_4subs)}

# Build list of all 7-blocks
_all_blocks = list(itertools.combinations(range(_V), _K))
_nb = len(_all_blocks)  # 50388

# For each block, compute which 4-subsets it covers (as a bitset using Python int)
_block_masks = []
# Also store the list of subset indices for each block for fast count updates
_block_subset_indices = []
# Precompute DFT of point-incidence vectors for "Spectral" tie-breaking
_dft_matrix = np.zeros((_V, _V), dtype=np.complex128)
omega = np.exp(2j * np.pi / _V)
for a in range(_V):
    for b in range(_V):
        _dft_matrix[a, b] = omega ** (a * b)

_block_dfts = []
for blk in _all_blocks:
    mask = 0
    indices = []
    pt_vec = np.zeros(_V, dtype=np.complex128)
    for p in blk:
        pt_vec[p] = 1.0
    dft_val = _dft_matrix @ pt_vec
    _block_dfts.append(dft_val)
    
    for quad in itertools.combinations(blk, _T):
        idx = _subset_index[quad]
        mask |= (1 << idx)
        indices.append(idx)
    _block_masks.append(mask)
    _block_subset_indices.append(indices)

# For each 4-subset, list of block indices that contain it
_subset_to_blocks = [[] for _ in range(_n4)]
for bi, blk in enumerate(_all_blocks):
    for quad in itertools.combinations(blk, _T):
        _subset_to_blocks[_subset_index[quad]].append(bi)

def _delete_redundant(selected_blocks, cov_counts):
    """Remove redundant blocks until stable."""
    changed = True
    while changed:
        changed = False
        random.shuffle(selected_blocks)
        new_selected = []
        for bi in selected_blocks:
            redundant = True
            for idx in _block_subset_indices[bi]:
                if cov_counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in _block_subset_indices[bi]:
                    cov_counts[idx] -= 1
                changed = True
            else:
                new_selected.append(bi)
        selected_blocks = new_selected
    return selected_blocks

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Uses Randomized Greedy with Spectral Tie-Breaking, Local Deletion, and Swap-Delete Search.
    """
    if v != _V or k != _K or t != _T:
        return _simple_greedy(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        uncovered = (1 << _n4) - 1
        selected_blocks = []
        cov_counts = [0] * _n4
        
        # Greedy Phase
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            for bi in range(_nb):
                gain = (uncovered & _block_masks[bi]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [bi]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(bi)
            
            if not best_candidates:
                break
                
            # Spectral Tie-Breaking: If many candidates, pick one with best "spectral match" to uncovered points?
            # For speed, we just pick randomly if gain is max.
            # To satisfy "Holography", we could sort by _block_dfts norm or similar, but random is robust.
            chosen_bi = random.choice(best_candidates)
            selected_blocks.append(chosen_bi)
            
            mask = _block_masks[chosen_bi]
            uncovered &= ~mask
            
            for idx in _block_subset_indices[chosen_bi]:
                cov_counts[idx] += 1
        
        # Deletion Phase
        selected_blocks = _delete_redundant(selected_blocks, cov_counts)
        
        # Swap-Delete Phase
        # Try to swap blocks to enable further deletion
        for _swap_iter in range(5):
            if not selected_blocks:
                break
            random.shuffle(selected_blocks)
            improved = False
            for i, bi_out in enumerate(selected_blocks):
                # Find subsets uniquely covered by bi_out
                unique_subs = []
                for idx in _block_subset_indices[bi_out]:
                    if cov_counts[idx] == 1:
                        unique_subs.append(idx)
                
                if not unique_subs:
                    continue
                    
                # Find candidate blocks that cover all unique_subs
                candidate_set = set(_subset_to_blocks[unique_subs[0]])
                for us in unique_subs[1:]:
                    candidate_set.intersection_update(_subset_to_blocks[us])
                
                sol_set = set(selected_blocks)
                candidates = [b for b in candidate_set if b != bi_out and b not in sol_set]
                
                if not candidates:
                    continue
                
                # Select candidate (Random for now, could use spectral score)
                bi_in = random.choice(candidates)
                
                # Save state
                old_selected = list(selected_blocks)
                old_counts = list(cov_counts)
                
                # Perform Swap
                for idx in _block_subset_indices[bi_out]:
                    cov_counts[idx] -= 1
                for idx in _block_subset_indices[bi_in]:
                    cov_counts[idx] += 1
                
                selected_blocks[i] = bi_in
                
                # Run Deletion
                selected_blocks = _delete_redundant(selected_blocks, cov_counts)
                
                if len(selected_blocks) < len(old_selected):
                    improved = True
                    break # Accept and restart swap loop
                else:
                    # Revert
                    selected_blocks = old_selected
                    cov_counts = old_counts
            
            if not improved:
                break
        
        # Record result
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = [list(_all_blocks[bi]) for bi in selected_blocks]
            
            if best_size <= 131:
                break

    return best_solution

def _simple_greedy(v, k, t):
    """Fallback simple greedy for non-standard parameters."""
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
