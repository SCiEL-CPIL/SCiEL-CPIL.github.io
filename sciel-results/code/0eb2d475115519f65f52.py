import itertools
import random
import time
import math
import numpy as np
import scipy.sparse as sp

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks/indices
all_blocks = []
block_subset_indices = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    idxs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        idxs.append(idx)
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    block_subset_indices.append(idxs)

N_BLOCKS = len(all_blocks)

# Precompute as a numpy array with -1 padding for blocks with fewer subsets
max_subsets_per_block = max(len(x) for x in block_subset_indices)
block_indices_np = np.full((N_BLOCKS, max_subsets_per_block), -1, dtype=np.int32)
block_lengths = np.array([len(x) for x in block_subset_indices], dtype=np.int32)
for i, indices in enumerate(block_subset_indices):
    if indices:
        block_indices_np[i, :len(indices)] = indices

# Precompute block masks as Python ints for fast bit operations (optional, maybe not needed if using counts)
# But useful for quick "does this block cover this subset" checks?
# Let's stick to indices and counts.

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
    deadline = start_time + 115  # Safety margin
    
    # State variables
    # current_blocks: list of block indices
    # counts: array of length N_SUBSETS, counts[s] = number of blocks in current_blocks covering s
    # current_set: set of block indices for O(1) membership check
    
    def greedy_init():
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        current_blocks = []
        current_set = set()
        uncovered = np.ones(N_SUBSETS, dtype=bool)
        n_uncovered = N_SUBSETS
        
        while n_uncovered > 0:
            # Compute gain for each block: number of uncovered subsets it covers
            # gains[b] = sum(uncovered[s] for s in block_subset_indices[b])
            # Vectorized:
            # valid_indices = block_indices_np >= 0
            # We can't easily do this with a single matrix multiply because of the -1 padding.
            # But we can precompute a sparse matrix.
            
            # Let's use the sparse matrix approach for speed.
            # A is N_BLOCKS x N_SUBSETS
            # gains = A @ uncovered (sparse-dense mat-vec)
            
            # To avoid building A every time (it's constant), we build it once globally?
            # No, it's constant. Let's build it outside construct?
            # It's large: 77520 x 3876. ~300MB if dense. Sparse is fine.
            
            pass # Placeholder, will use global sparse matrix

    # Build Sparse Incidence Matrix A (N_BLOCKS x N_SUBSETS)
    # A[b, s] = 1 if block b covers subset s
    row_ind = []
    col_ind = []
    for b_idx, indices in enumerate(block_subset_indices):
        for s_idx in indices:
            row_ind.append(b_idx)
            col_ind.append(s_idx)
    
    A = sp.csr_matrix((np.ones(len(row_ind), dtype=np.int8), (row_ind, col_ind)), shape=(N_BLOCKS, N_SUBSETS))
    
    # Greedy Initialization
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    current_set = set()
    uncovered = np.ones(N_SUBSETS, dtype=bool)
    n_uncovered = N_SUBSETS
    
    # We need to update A? No, A is static.
    # But we need to avoid picking the same block twice.
    # We can set A row to 0 when picked? Or just check `current_set`.
    # Checking `current_set` in the argmax loop is slow.
    # Better: maintain a `active` mask or modify A?
    # Modifying A (sparse) is expensive.
    # Alternative: Keep A static. Compute gains. Set gains of already picked blocks to -inf.
    
    while n_uncovered > 0:
        gains = A @ uncovered  # shape (N_BLOCKS,)
        # Mask out already selected blocks
        for b in current_blocks:
            gains[b] = -1
        best_idx = int(np.argmax(gains))
        if gains[best_idx] <= 0:
            break
        
        # Add block
        current_blocks.append(best_idx)
        current_set.add(best_idx)
        
        # Update counts and uncovered
        for s_idx in block_subset_indices[best_idx]:
            counts[s_idx] += 1
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                n_uncovered -= 1
        
    # Function to prune redundant blocks
    def prune(blocks, counts, block_set):
        changed = True
        while changed:
            changed = False
            # Iterate backwards to allow popping
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for s_idx in block_subset_indices[b]:
                    if counts[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block b
                    for s_idx in block_subset_indices[b]:
                        counts[s_idx] -= 1
                    block_set.discard(b)
                    blocks.pop(i)
                    changed = True
        return blocks, counts, block_set

    # Initial Prune
    current_blocks, counts, current_set = prune(current_blocks, counts, current_set)
    best_blocks = current_blocks.copy()
    best_counts = counts.copy()
    best_set = set(current_set)
    best_len = len(best_blocks)
    
    # Iterated Local Search
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Kick: Remove R random blocks
        R = 2  # Number of blocks to remove for perturbation
        if len(current_blocks) < R:
            break
            
        kick_indices = random.sample(range(len(current_blocks)), R)
        kick_indices.sort(reverse=True) # Remove from end first to keep indices valid
        
        for i in kick_indices:
            b = current_blocks[i]
            for s_idx in block_subset_indices[b]:
                counts[s_idx] -= 1
            current_set.discard(b)
            current_blocks.pop(i)
            
        # Identify uncovered subsets (holes)
        uncovered = (counts == 0).astype(bool)
        n_uncovered = np.sum(uncovered)
        
        # Repair: Greedily add blocks to cover holes
        # We only care about covering the holes.
        # Gain for a block is the number of holes it covers.
        # We can restrict A to the columns corresponding to holes?
        # Or just compute gains = A @ uncovered.
        
        while n_uncovered > 0:
            gains = A @ uncovered
            # Mask out already selected blocks
            for b in current_set:
                gains[b] = -1
            best_idx = int(np.argmax(gains))
            
            if gains[best_idx] <= 0:
                # Cannot repair? This shouldn't happen if we have a valid covering, 
                # but removing blocks might create a state that is hard to repair with just 1 block at a time?
                # Actually, since the original set was a covering, and we only removed a few, 
                # the holes are limited. There must be blocks covering them.
                # If gains <= 0, it means no block covers any hole? Impossible.
                # Unless we masked out the only blocks that cover the holes.
                # If this happens, we break and restart from best.
                break
                
            # Add block
            current_blocks.append(best_idx)
            current_set.add(best_idx)
            
            for s_idx in block_subset_indices[best_idx]:
                counts[s_idx] += 1
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    n_uncovered -= 1
        
        if n_uncovered > 0:
            # Repair failed, revert to best
            current_blocks = best_blocks.copy()
            counts = best_counts.copy()
            current_set = set(best_set)
            continue

        # Prune again
        current_blocks, counts, current_set = prune(current_blocks, counts, current_set)
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks.copy()
            best_counts = counts.copy()
            best_set = set(current_set)
            no_improve = 0
        else:
            # Accept if equal? To allow wandering.
            if len(current_blocks) == best_len:
                # Accept with some probability to escape local minima?
                # Or just keep the current one if it's different?
                # Let's accept if it's different to explore.
                if current_set != best_set:
                    best_blocks = current_blocks.copy()
                    best_counts = counts.copy()
                    best_set = set(current_set)
                no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                # Revert to best and do a stronger kick?
                current_blocks = best_blocks.copy()
                counts = best_counts.copy()
                current_set = set(best_set)
                no_improve = 0

    # Final Prune on best
    final_blocks, final_counts, final_set = prune(best_blocks, best_counts, set(best_set))
    
    return [list(all_blocks[i]) for i in final_blocks]
