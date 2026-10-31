<<<<<<< SEARCH
import time
import random
import itertools

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# Precompute all t-subsets and each block's coverage as an integer bitmask.
_T_SUBSETS = []
_T_INDEX = {}
_BLOCK_COV = {}
_CACHE = {}


def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    index = {sub: i for i, sub in enumerate(t_subs)}
    n = len(t_subs)
    cov = {}
    for blk in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= (1 << index[sub])
        cov[blk] = m
    _CACHE[key] = (t_subs, index, n, cov)
    return _CACHE[key]
=======
import time
import random
import itertools

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# Precompute all t-subsets and each block's coverage as an integer bitmask.
_T_SUBSETS = []
_T_INDEX = {}
_BLOCK_COV = {}
_CACHE = {}


def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    index = {sub: i for i, sub in enumerate(t_subs)}
    n = len(t_subs)
    all_blocks = list(itertools.combinations(range(v), k))
    cov = {}
    for blk in all_blocks:
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= (1 << index[sub])
        cov[blk] = m
    
    # Prepare numpy matrix for fast greedy
    if _HAS_NUMPY:
        cov_matrix = np.zeros((len(all_blocks), n), dtype=np.uint8)
        for i, blk in enumerate(all_blocks):
            m = cov[blk]
            # Extract set bits and set in matrix
            # This is faster than iterating all n for each block if n is large, 
            # but here n=3876 and bits per block=35.
            # So we iterate the 35 bits.
            j = 0
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                cov_matrix[i, idx] = 1
                m &= m - 1
    else:
        cov_matrix = None

    _CACHE[key] = (t_subs, index, n, cov, all_blocks, cov_matrix)
    return _CACHE[key]
>>>>>>> REPLACE

Now update `_construct_direct` to use the matrix.

