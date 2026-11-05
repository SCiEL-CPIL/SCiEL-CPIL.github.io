<<<<<<< SEARCH
import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Coverage matrix
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

weights_inv_freq = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)
=======
import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

all_blocks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS
block_bit_indices = []

for block in itertools.combinations(range(V), K):
    block_idx = len(all_blocks)
    bits = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        bits.append(idx)
        blocks_containing_subset[idx].append(block_idx)
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    block_bit_indices.append(bits)

N_BLOCKS = len(all_blocks)

weights_inv_freq = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)
>>>>>>> REPLACE

