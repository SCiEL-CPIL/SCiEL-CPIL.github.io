import numpy as np
import itertools
import time
import random
from collections import defaultdict
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Pre-compute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Pre-compute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # Build sparse incidence matrix A: A[block_idx, t_subset_idx] = 1
    rows = []
    cols = []
    for bi, block in enumerate(k_blocks):
        for subset in itertools.combinations(block, t):
            t_idx = t_subset_to_idx[subset]
            rows.append(bi)
            cols.append(t_idx)
            
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(num_k_blocks, num_t_subsets))
    
    def time_left():
        return deadline - time.time()
    
    def greedy_cover(random_seed=None):
        """Vectorized Greedy Set Cover."""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        used = np.zeros(num_k_blocks, dtype=bool)
        selected = []
        
        while time_left() > 1.0:
            # If all covered, break
            if np.all(coverage > 0):
                break
                
            # Compute gains: number of uncovered t-subsets covered by each block
            # A.dot(vector) works with sparse matrix
            uncovered_mask = (coverage == 0).astype(np.int32)
            gains = A.dot(uncovered_mask)
            
            # Mask out used blocks
            gains[used] = -1
            
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                break
            
            selected.append(best_idx)
            used[best_idx] = True
            
            # Update coverage
            # A[best_idx] is a sparse row. Convert to dense for addition to dense coverage
            # This is O(C(k,t)) which is small (35 for 7,4)
            coverage += A[best_idx].toarray().flatten()
            
        return selected

    def remove_redundant(block_indices):
        """Remove blocks that are redundant."""
        if not block_indices:
            return []
        
        # Count coverage from selected blocks only
        # We can sum the rows of A corresponding to block_indices
        if len(block_indices) == 0:
             return []
             
        # Efficient way: 
        # coverage = np.zeros(num_t_subsets)
        # for bi in block_indices:
        #    coverage += A[bi].toarray().flatten()
        
        # Using sparse matrix slicing might be faster if we keep a sparse coverage?
        # But num_t_subsets is small (3876). Dense is fine.
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for bi in block_indices:
            coverage += A[bi].toarray().flatten()
            
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            new_result = []
            for bi in result:
                # Check if redundant: all t-subsets in this block are covered by >= 2 blocks
                # i.e., coverage[t_idx] > 1 for all t_idx in A[bi]
                t_idxs = A[bi].indices
                # If any t_idx has coverage 1, it's not redundant
                if np.any(coverage[t_idxs] <= 1):
                    new_result.append(bi)
                else:
                    # Redundant, remove
                    coverage -= A[bi].toarray().flatten()
                    changed = True
            result = new_result
        return result

    def local_search(block_indices, max_iters=50):
        """
        Simple Local Search:
        1. Try to remove each block. If valid, remove.
        2. Try to swap a block with another? (Might be too slow)
        Let's stick to aggressive removal first.
        """
        if not block_indices:
            return block_indices
            
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for bi in block_indices:
            coverage += A[bi].toarray().flatten()
            
        current_blocks = list(block_indices)
        
        # Phase 1: Remove redundant
        current_blocks = remove_redundant(current_blocks)
        
        # Phase 2: Try to replace blocks?
        # For time efficiency, let's just do multiple passes of removal.
        
        return current_blocks

    best_solution = None
    best_len = float('inf')
    
    # Run multiple restarts with different random seeds to break ties differently
    # The vectorized greedy is deterministic unless we add randomness to tie-breaking or initial state.
    # We can add random noise to gains to simulate different paths.
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time_left() < 5.0:
            break
            
        # Add small random noise to gains to vary the path
        # We modify the greedy function slightly to accept noise?
        # Or just rely on the fact that if we start with a random subset?
        
        # Alternative: Randomized Greedy
        # Pick a random block from the top 5% of gains?
        
        # Let's implement a randomized greedy inside the loop
        selected = []
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        used = np.zeros(num_k_blocks, dtype=bool)
        
        while time_left() > 1.0:
            if np.all(coverage > 0):
                break
                
            uncovered_mask = (coverage == 0).astype(np.int32)
            gains = A.dot(uncovered_mask)
            gains[used] = -1
            
            # Randomized selection:
            # Find max gain
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            # Get all blocks with max gain (or close to it)
            # To avoid O(N) scan, we can just pick the argmax.
            # To randomize, we can pick randomly from blocks with gain > 0.9 * max_gain
            threshold = 0.9 * max_gain
            candidates = np.where((gains >= threshold) & (gains > 0))[0]
            
            if len(candidates) == 0:
                candidates = np.array([np.argmax(gains)])
                
            best_idx = np.random.choice(candidates)
            
            selected.append(best_idx)
            used[best_idx] = True
            coverage += A[best_idx].toarray().flatten()
            
        # Post-process
        selected = remove_redundant(selected)
        
        # Local search (just removal for now)
        # Try to remove blocks one by one in random order
        random.shuffle(selected)
        i = 0
        while i < len(selected) and time_left() > 1.0:
            bi = selected[i]
            # Check if redundant
            t_idxs = A[bi].indices
            if np.all(coverage[t_idxs] > 1):
                # Remove
                coverage -= A[bi].toarray().flatten()
                selected.pop(i)
            else:
                i += 1
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_solution = list(selected)
            
            if best_len <= 151:
                break

    if best_solution is None:
        # Fallback to deterministic greedy
        best_solution = greedy_cover()
        best_solution = remove_redundant(best_solution)

    # Final deduplication (should be unique by index, but just in case)
    seen = set()
    unique_blocks = []
    for bi in best_solution:
        if bi not in seen:
            seen.add(bi)
            unique_blocks.append(bi)
            
    result = [list(k_blocks[bi]) for bi in unique_blocks]
    
    return result
