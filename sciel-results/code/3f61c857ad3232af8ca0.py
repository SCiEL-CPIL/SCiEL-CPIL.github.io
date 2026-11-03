import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build sparse matrix A of size n_k x n_t
    # A[i, j] = 1 if t_subs[j] is subset of k_subs[i]
    row_indices = []
    col_indices = []
    data = []
    
    # For each block, find which t-subsets it contains
    for i, block in enumerate(k_subs):
        block_set = set(block)
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
                
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)
    
    # Pre-compute the column indices for each block (for fast access)
    block_cols = []
    for i in range(n_k):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        cols = A.indices[row_start:row_end]
        block_cols.append(cols)
    
    # Build bitset representation for fast coverage checks
    # Each block is represented as a bitset of t-subsets it covers
    # We'll use numpy arrays for efficient operations
    n_bits_per_block = n_t
    # Use int64 arrays for bitsets
    n_words = (n_t + 63) // 64
    block_bitsets = np.zeros((n_k, n_words), dtype=np.uint64)
    
    for i, block in enumerate(k_subs):
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                word_idx = j // 64
                bit_idx = j % 64
                block_bitsets[i, word_idx] |= (1 << bit_idx)
    
    def greedy_cover(seed_blocks=None, max_iter=None):
        """Greedy set cover with optional seed blocks."""
        u = np.ones(n_t, dtype=np.int8)
        selected_blocks = []
        
        # If we have seed blocks, start with them
        if seed_blocks:
            for idx in seed_blocks:
                cols = block_cols[idx]
                u[cols] = 0
                selected_blocks.append(idx)
        
        if max_iter is None:
            max_iter = 10000
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if u.sum() == 0:
                break
                
            # Calculate gain for each block
            gains = A @ u
            # Find best block
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
                
            selected_blocks.append(best_idx)
            
            # Update uncovered vector
            cols = block_cols[best_idx]
            u[cols] = 0
            
        return selected_blocks
    
    def remove_redundant(selected_blocks):
        """Remove redundant blocks from the covering."""
        if not selected_blocks:
            return selected_blocks
            
        # Pre-compute lists of t-subset indices for each selected block
        selected_t_subs = []
        for idx in selected_blocks:
            cols = block_cols[idx]
            selected_t_subs.append(cols)
            
        # Initialize cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for cols in selected_t_subs:
            cover_count[cols] += 1
            
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(selected_blocks)):
                cols = selected_t_subs[i]
                # Check if any col in cols has cover_count == 1
                is_essential = False
                for c in cols:
                    if cover_count[c] == 1:
                        is_essential = True
                        break
                    
                if not is_essential:
                    to_remove.append(i)
                    
            if to_remove:
                changed = True
                # Remove blocks from the end to maintain indices
                for i in sorted(to_remove, reverse=True):
                    # Update cover_count
                    cols = selected_t_subs[i]
                    cover_count[cols] -= 1
                    # Remove from lists
                    selected_blocks.pop(i)
                    selected_t_subs.pop(i)
                    
        return selected_blocks
    
    def benders_improve(selected_blocks, max_rounds=10):
        """Benders-style improvement: use bitset coverage check to find better blocks."""
        if not selected_blocks:
            return selected_blocks
            
        best = list(selected_blocks)
        best_size = len(best)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Compute current coverage as bitset
            coverage = np.zeros(n_words, dtype=np.uint64)
            for idx in best:
                coverage |= block_bitsets[idx]
            
            # Check if fully covered
            uncovered_mask = ~coverage
            # Handle the last word which may have extra bits
            if n_t % 64 != 0:
                extra_bits = 64 - (n_t % 64)
                uncovered_mask[-1] &= (1 << extra_bits) - 1
            
            if uncovered_mask.any() == 0:
                break
            
            # Find t-subsets that are uncovered
            uncovered_indices = []
            for w in range(n_words):
                if uncovered_mask[w]:
                    word = uncovered_mask[w]
                    while word:
                        bit = (word & -word).bit_length() - 1
                        idx = w * 64 + bit
                        if idx < n_t:
                            uncovered_indices.append(idx)
                        word &= word - 1
            
            if not uncovered_indices:
                break
            
            # Try to find a single block that covers the most uncovered t-subsets
            # and can replace one of the existing blocks
            best_improvement = None
            best_new_size = best_size
            
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                
                # Temporarily remove block i
                temp_coverage = coverage & ~block_bitsets[best[i]]
                
                # Find which t-subsets become uncovered
                newly_uncovered = temp_coverage ^ coverage
                # Actually, we want t-subsets that were covered by block i but not by others
                # newly_uncovered should be: coverage without block i, then find what's missing
                # Let's recompute: what t-subsets are covered by others only
                others_coverage = np.zeros(n_words, dtype=np.uint64)
                for j, idx in enumerate(best):
                    if j != i:
                        others_coverage |= block_bitsets[idx]
                
                # T-subsets that become uncovered if we remove block i
                would_be_uncovered = coverage & ~others_coverage
                
                if not would_be_uncovered.any():
                    # Block i is redundant, just remove it
                    new_best = best[:i] + best[i+1:]
                    if len(new_best) < best_new_size:
                        best_new_size = len(new_best)
                        best_improvement = new_best
                    continue
                
                # Find a replacement block that covers all of would_be_uncovered
                # and has minimal overlap with others_coverage (to avoid redundancy)
                best_candidate = None
                best_candidate_score = -1
                
                # Sample a subset of candidates for speed
                sample_size = min(500, n_k)
                candidates = random.sample(range(n_k), sample_size)
                
                for cand in candidates:
                    if cand == best[i]:
                        continue
                    
                    cand_bits = block_bitsets[cand]
                    
                    # Check if candidate covers all of would_be_uncovered
                    if not (cand_bits & would_be_uncovered).all():
                        # Need to check bit by bit if would_be_uncovered is not all 1s
                        # For simplicity, check if candidate covers the same or more
                        if (cand_bits | would_be_uncovered) != would_be_uncovered:
                            continue
                    
                    # Score: number of new t-subsets covered (not in others_coverage)
                    new_coverage = cand_bits & ~others_coverage
                    score = new_coverage.sum()
                    
                    # Prefer blocks that cover more unique t-subsets
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = cand
                
                if best_candidate is not None:
                    new_best = best[:i] + [best_candidate] + best[i+1:]
                    # Check if this actually reduces size or is same size with better coverage
                    if len(new_best) < best_new_size:
                        best_new_size = len(new_best)
                        best_improvement = new_best
                    elif len(new_best) == best_new_size:
                        # Same size, but check if coverage is better
                        new_cov = np.zeros(n_words, dtype=np.uint64)
                        for idx in new_best:
                            new_cov |= block_bitsets[idx]
                        old_cov = coverage
                        # Count how many t-subsets are covered
                        new_count = 0
                        for w in range(n_words):
                            new_count += bin(new_cov[w]).count('1')
                        old_count = 0
                        for w in range(n_words):
                            old_count += bin(old_cov[w]).count('1')
                        if new_count > old_count:
                            best_improvement = new_best
        
            if best_improvement is not None:
                best = best_improvement
                best_size = len(best)
            else:
                break
        
        return best
    
    # Run greedy with different random seeds
    best_result = None
    best_size = float('inf')
    
    # First, run standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)
    
    # Try randomized restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Create a random seed
        # Pick a few random blocks to start with
        seed_size = min(5, n_k)
        seed_blocks = random.sample(range(n_k), seed_size)
        
        result = greedy_cover(seed_blocks=seed_blocks)
        result = remove_redundant(result)
        
        if len(result) < best_size:
            best_result = result
            best_size = len(result)
    
    # Try Benders-style improvement on the best result
    if best_result and time.time() < deadline - 15:
        improved = benders_improve(best_result, max_rounds=5)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)
    
    # Convert indices back to blocks
    if best_result is None:
        best_result = []
        
    final_blocks = [list(k_subs[i]) for i in best_result]
    
    return final_blocks
