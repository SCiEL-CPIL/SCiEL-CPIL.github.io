# Precompute block masks for stale check
_block_masks = []
for blk in _all_blocks:
    mask = 0
    for quad in itertools.combinations(blk, _T):
        idx = _subset_index[quad]
        mask |= (1 << idx)
    _block_masks.append(mask)
