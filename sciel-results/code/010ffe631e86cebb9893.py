<<<<<<< SEARCH
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
=======
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

# --- Combinatorial Holography Precomputation ---
# 1. Orbit IDs based on cyclic shift in Z19
# Normalize a 4-subset by shifting the smallest element to 0
def _get_orbit_id(subset):
    # subset is a tuple of 4 sorted ints
    min_val = subset[0]
    shifted = tuple((x - min_val) % _V for x in subset)
    return shifted

# Map each 4-subset to its orbit representative
_orbit_reps = []
_orbit_id_map = {}
for s in _all_4subs:
    rep = _get_orbit_id(s)
    if rep not in _orbit_id_map:
        _orbit_id_map[rep] = len(_orbit_id_map)
    _orbit_reps.append(_orbit_id_map[rep])

_n_orbits = len(_orbit_id_map)

# 2. For each block, compute which 4-subsets it covers (as a bitset using Python int)
_block_masks = []
# Also store the list of subset indices for each block for fast count updates
_block_subset_indices = []
# Store orbit counts for each block (how many of its 4-subsets belong to each orbit)
# This is used for the spectral weighting
_block_orbit_counts = []

for blk in _all_blocks:
    mask = 0
    indices = []
    orbit_counts = [0] * _n_orbits
    for quad in itertools.combinations(blk, _T):
        idx = _subset_index[quad]
        mask |= (1 << idx)
        indices.append(idx)
        o_id = _orbit_reps[idx]
        orbit_counts[o_id] += 1
    _block_masks.append(mask)
    _block_subset_indices.append(indices)
    _block_orbit_counts.append(orbit_counts)

# For each 4-subset, list of block indices that contain it
_subset_to_blocks = [[] for _ in range(_n4)]
for bi, blk in enumerate(_all_blocks):
    for quad in itertools.combinations(blk, _T):
        _subset_to_blocks[_subset_index[quad]].append(bi)

# FFT components kept for "Combinatorial Holography" flavor
omega = np.exp(2j * np.pi / _V)
_dft_matrix = np.zeros((_V, _V), dtype=np.complex128)
for a in range(_V):
    for b in range(_V):
        _dft_matrix[a, b] = omega ** (a * b)
>>>>>>> REPLACE

