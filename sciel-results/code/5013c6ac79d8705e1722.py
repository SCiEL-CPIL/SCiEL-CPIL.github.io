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
    
    # Precompute blocks covering each t-subset for fast intersection in local search
    # blocks_covering[t_idx] = list of block indices
    blocks_covering = [[] for _ in range(num_t_subsets)]
    # We can derive this from A or the loop above
    # To save time, let's just use A's structure or rebuild if needed.
    # Actually, A is CSR, so A.indices gives columns for a row.
    # We need the reverse: for each column, which rows?
    # A.tocsc() would give this.
    A_csc = A.tocsc()
    for t_idx in range(num_t_subsets):
        blocks_covering[t_idx] = A_csc.indices[A_csc.indptr[t_idx]:A_csc.indptr[t_idx+1]].tolist()

    def time_left():
        return deadline - time.time()
    
    def greedy_cover(random_seed=None):
        """Vectorized Greedy Set Cover with randomization."""
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
            uncovered_mask = (coverage == 0).astype(np.int32)
            gains = A.dot(uncovered_mask)
            
            # Mask out used blocks
            gains[used] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            # Randomized selection from top candidates
            # Threshold: 90% of max gain
            threshold = 0.9 * max_gain
            candidates = np.where((gains >= threshold) & (gains > 0))[0]
            
            if len(candidates) == 0:
                best_idx = np.argmax(gains)
            else:
                best_idx = np.random.choice(candidates)
            
            selected.append(best_idx)
            used[best_idx] = True
            coverage += A[best_idx].toarray().flatten()
            
        return selected

    def remove_redundant(block_indices):
        """Remove blocks that are redundant."""
        if not block_indices:
            return []
            
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for bi in block_indices:
            coverage += A[bi].toarray().flatten()
            
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            new_result = []
            for bi in result:
                t_idxs = A[bi].indices
                if np.any(coverage[t_idxs] <= 1):
                    new_result.append(bi)
                else:
                    coverage -= A[bi].toarray().flatten()
                    changed = True
            result = new_result
        return result

    def local_search_with_swaps(block_indices, max_passes=5):
        """
        1. Remove redundant blocks.
        2. Try 1-for-1 swaps to improve solution structure.
        """
        if not block_indices:
            return block_indices
            
        # Phase 1: Aggressive Removal
        current_blocks = remove_redundant(block_indices)
        
        # Phase 2: 1-for-1 Swaps
        # We maintain coverage counts
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        for bi in current_blocks:
            coverage += A[bi].toarray().flatten()
        
        # Set of current block indices for O(1) lookup
        current_set = set(current_blocks)
        
        for _ in range(max_passes):
            if time_left() < 2.0:
                break
                
            improved = False
            # Shuffle to break symmetry
            shuffled_blocks = current_blocks[:]
            random.shuffle(shuffled_blocks)
            
            for bi in shuffled_blocks:
                if time_left() < 2.0:
                    break
                    
                # Try removing bi
                # Check if any t-subset in bi is covered only by bi
                t_idxs = A[bi].indices
                if np.any(coverage[t_idxs] == 1):
                    # Cannot remove bi alone. Try to swap.
                    # Find uncovered t-subsets if we remove bi
                    # These are the t_idxs where coverage == 1
                    uncovered = t_idxs[coverage[t_idxs] == 1]
                    
                    if len(uncovered) == 0:
                        continue # Should not happen if logic above is correct, but safety
                    
                    # Find a block that covers ALL uncovered t-subsets
                    # Intersection of blocks_covering[u] for u in uncovered
                    # Sort uncovered by size of blocks_covering to speed up intersection
                    uncovered_sorted = sorted(uncovered, key=lambda u: len(blocks_covering[u]))
                    
                    # Start with the smallest set
                    candidates = set(blocks_covering[uncovered_sorted[0]])
                    
                    # Intersect with others
                    for u in uncovered_sorted[1:]:
                        candidates.intersection_update(blocks_covering[u])
                        if not candidates:
                            break
                    
                    # Filter out current blocks
                    candidates.difference_update(current_set)
                    
                    if candidates:
                        # Pick a random candidate
                        new_bi = random.choice(list(candidates))
                        
                        # Perform swap
                        # Remove old
                        coverage -= A[bi].toarray().flatten()
                        current_set.remove(bi)
                        current_blocks.remove(bi)
                        
                        # Add new
                        coverage += A[new_bi].toarray().flatten()
                        current_set.add(new_bi)
                        current_blocks.append(new_bi)
                        
                        improved = True
                else:
                    # Can remove bi
                    coverage -= A[bi].toarray().flatten()
                    current_set.remove(bi)
                    current_blocks.remove(bi)
                    improved = True
            
            if not improved:
                break
                
        return current_blocks

    best_solution = None
    best_len = float('inf')
    
    num_restarts = 25
    for restart in range(num_restarts):
        if time_left() < 5.0:
            break
            
        # Randomized Greedy
        selected = greedy_cover(random_seed=restart * 1000 + 42)
        
        # Local Search with Swaps
        selected = local_search_with_swaps(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_solution = list(selected)
            
            if best_len <= 151:
                break

    if best_solution is None:
        # Fallback
        best_solution = greedy_cover()
        best_solution = remove_redundant(best_solution)

    # Final deduplication
    seen = set()
    unique_blocks = []
    for bi in best_solution:
        if bi not in seen:
            seen.add(bi)
            unique_blocks.append(bi)
            
    result = [list(k_blocks[bi]) for bi in unique_blocks]
    
    return result
