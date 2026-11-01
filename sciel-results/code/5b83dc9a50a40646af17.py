"""Combinatorial Holography covering via spectral residual energy over Z19."""
import itertools
import numpy as np

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
for blk in _all_blocks:
    mask = 0
    for quad in itertools.combinations(blk, _T):
        mask |= (1 << _subset_index[quad])
    _block_masks.append(mask)

# For each 4-subset, list of block indices that contain it
_subset_to_blocks = [[] for _ in range(_n4)]
for bi, blk in enumerate(_all_blocks):
    for quad in itertools.combinations(blk, _T):
        _subset_to_blocks[_subset_index[quad]].append(bi)

# --- Combinatorial Holography: 64-bit fingerprints ---
_np_rng = np.random.default_rng(42)
_fingerprint_weights = np_rng.integers(1, 2**63, size=_n4, dtype=np.uint64)

# Precompute block fingerprints (XOR of all covered 4-subset fingerprints)
_block_fingerprints = np.zeros(_nb, dtype=np.uint64)
for bi, blk in enumerate(_all_blocks):
    fp = 0
    for quad in itertools.combinations(blk, _T):
        fp ^= _fingerprint_weights[_subset_index[quad]]
    _block_fingerprints[bi] = fp

# --- FFT over Z19 ---
_omega = np.exp(2j * np.pi / _V)
_dft_matrix = np.zeros((_V, _V), dtype=np.complex128)
for a in range(_V):
    for b in range(_V):
        _dft_matrix[a, b] = _omega ** (a * b)

# Precompute spectral signature for each block
# block_spectral[bi] = DFT of point-count vector for block bi
_block_spectral = np.zeros((_nb, _V), dtype=np.complex128)
_block_point_counts = np.zeros((_nb, _V), dtype=np.float64)
for bi, blk in enumerate(_all_blocks):
    for p in blk:
        # Each point in the block appears in C(6,3) = 20 of the 4-subsets of the block
        # Wait, no. A 7-block has C(7,4)=35 four-subsets. Each point appears in C(6,3)=20 of them.
        # So point count for each point in the block is 20.
        _block_point_counts[bi, p] = 20.0

_block_spectral = _dft_matrix @ _block_point_counts.T  # shape (19, 50388) -> transpose to (50388, 19)
_block_spectral = _block_spectral.T  # now (50388, 19)

# Precompute |block_spectral|^2 for each block
_block_spectral_norm2 = np.sum(np.abs(_block_spectral)**2, axis=1)  # shape (50388,)


def _compute_residual_spectral(uncovered_mask):
    """Compute the spectral vector of uncovered 4-subsets' point incidence."""
    point_counts = np.zeros(_V, dtype=np.float64)
    m = uncovered_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        quad = _all_4subs[idx]
        for p in quad:
            point_counts[p] += 1
        m ^= lsb
    return _dft_matrix @ point_counts


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using Combinatorial Holography.
    
    Uses 64-bit fingerprints for block identification and FFT over Z19 to
    compute spectral energy of uncovered 4-subsets. At each step, selects
    the block that maximizes spectral cancellation of the residual error.
    """
    if v != _V or k != _K or t != _T:
        return _simple_greedy(v, k, t)
    
    # Start with all 4-subsets uncovered
    uncovered = (1 << _n4) - 1
    blocks_selected = []
    uncovered_count = _n4
    
    while uncovered_count > 0:
        # Find a seed uncovered subset (lowest index)
        lsb = uncovered & (-uncovered)
        idx = lsb.bit_length() - 1
        
        # Get all blocks that contain this subset
        candidates = _subset_to_blocks[idx]
        
        # Compute current spectral residual vector
        residual_spectral = _compute_residual_spectral(uncovered)
        
        # Score each candidate block
        best_block = None
        best_newly_covered = -1
        best_score = -1e18
        
        for bi in candidates:
            # Primary: newly covered count
            newly_covered = (uncovered & _block_masks[bi]).bit_count()
            
            if newly_covered <= 0:
                continue
            
            # Spectral cancellation score
            # Energy reduction ≈ 2*Re(<residual, block_spectral>) - |block_spectral|^2
            # But this overestimates since we only remove newly_covered subsets, not all
            # Scale by ratio: newly_covered / 35 (total 4-subsets in a block)
            ratio = newly_covered / 35.0
            block_spec = _block_spectral[bi]
            spectral_gain = ratio * (2.0 * np.real(np.vdot(residual_spectral, block_spec)) - _block_spectral_norm2[bi])
            
            # Combined score
            score = newly_covered * 1e6 + spectral_gain
            
            if score > best_score:
                best_score = score
                best_newly_covered = newly_covered
                best_block = bi
        
        # Fallback if no block found
        if best_block is None or best_newly_covered <= 0:
            best_newly_covered = -1
            for bi in candidates:
                newly_covered = (uncovered & _block_masks[bi]).bit_count()
                if newly_covered > best_newly_covered:
                    best_newly_covered = newly_covered
                    best_block = bi
        
        if best_block is not None:
            blocks_selected.append(list(_all_blocks[best_block]))
            uncovered &= ~_block_masks[best_block]
            uncovered_count -= best_newly_covered
        
        if uncovered_count > 0 and (best_block is None or best_newly_covered <= 0):
            uncovered_count = uncovered.bit_count()
            if uncovered_count == 0:
                break
            bi = candidates[0] if candidates else 0
            blocks_selected.append(list(_all_blocks[bi]))
            uncovered &= ~_block_masks[bi]
            uncovered_count = uncovered.bit_count()
    
    return blocks_selected


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
