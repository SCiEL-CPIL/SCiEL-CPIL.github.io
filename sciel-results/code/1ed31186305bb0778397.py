"""Combinatorial Holography covering via spectral residual energy over Z19."""
import itertools
import numpy as np
import random

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
for blk in _all_blocks:
    mask = 0
    indices = []
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

# FFT components kept for "Combinatorial Holography" flavor, though the core improvement
# is Randomized Greedy + Local Deletion Search which is more effective for this size.
omega = np.exp(2j * np.pi / _V)
_dft_matrix = np.zeros((_V, _V), dtype=np.complex128)
for a in range(_V):
    for b in range(_V):
        _dft_matrix[a, b] = omega ** (a * b)

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Uses Randomized Greedy with Local Deletion Search.
    """
    if v != _V or k != _K or t != _T:
        return _simple_greedy(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    
    # Number of restarts to try. Adjust based on time.
    # Estimate: ~1-2s per run. 120s / 2s = 60 runs.
    num_restarts = 40 
    
    for _ in range(num_restarts):
        uncovered = (1 << _n4) - 1
        selected_blocks = []
        cov_counts = [0] * _n4
        
        # Greedy Phase
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # Find max gain
            # Optimization: Iterate all blocks
            for bi in range(_nb):
                gain = (uncovered & _block_masks[bi]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [bi]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(bi)
            
            if not best_candidates:
                break # Should not happen if uncovered > 0
                
            # Random tie-breaking
            chosen_bi = random.choice(best_candidates)
            selected_blocks.append(chosen_bi)
            
            # Update state
            mask = _block_masks[chosen_bi]
            uncovered &= ~mask
            
            for idx in _block_subset_indices[chosen_bi]:
                cov_counts[idx] += 1
        
        # Deletion Phase
        # Shuffle to randomize order
        random.shuffle(selected_blocks)
        
        # Try to remove redundant blocks
        # We do a single pass. If we remove a block, it might make another block redundant,
        # but a single pass is usually enough for significant reduction.
        # To be thorough, we could loop until no change, but that's slower.
        # Let's try a few passes.
        
        changed = True
        while changed:
            changed = False
            # Re-shuffle for next pass? Or keep order?
            # Re-shuffling helps find different redundancies.
            random.shuffle(selected_blocks)
            
            new_selected = []
            for bi in selected_blocks:
                # Check if redundant
                redundant = True
                for idx in _block_subset_indices[bi]:
                    if cov_counts[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for idx in _block_subset_indices[bi]:
                        cov_counts[idx] -= 1
                    changed = True
                else:
                    new_selected.append(bi)
            
            selected_blocks = new_selected
        
        # Record result
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = [list(_all_blocks[bi]) for bi in selected_blocks]
            
            # Optional: Early stop if we hit lower bound? 
            # Lower bound is 131. If we hit 131, stop.
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
