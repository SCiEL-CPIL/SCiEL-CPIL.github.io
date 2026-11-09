import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using vectorized global greedy
    multi-start followed by local search refinement.
    """
    deadline = time.time() + 145.0  # Safety margin

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    ksub_to_idx = {ks: i for i, ks in enumerate(all_ksubs)}
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    ksub_to_tsub_indices = {}
    for ks in all_ksubs:
        idxs = [tsub_to_idx[comb] for comb in itertools.combinations(ks, t)]
        ksub_to_tsub_indices[ks] = idxs
    
    # Build coverage matrix: cov_matrix[ksub_idx, tsub_idx] = 1 if ksub covers tsub
    # This is a sparse matrix of size n_ksubs x n_tsubs
    # For v=19, k=8, t=5: 75582 x 11628
    # Each k-subset covers C(8,5)=56 t-subsets
    cov_rows = []
    cov_cols = []
    for ki, ks in enumerate(all_ksubs):
        for ti in ksub_to_tsub_indices[ks]:
            cov_rows.append(ki)
            cov_cols.append(ti)
    
    from scipy.sparse import csr_matrix
    cov_matrix = csr_matrix((np.ones(len(cov_rows), dtype=np.int8), (cov_rows, cov_cols)), shape=(n_ksubs, n_tsubs))
    
    # Also build the transpose for fast "which ksubs cover a given tsub"
    cov_matrix_T = cov_matrix.T.tocsr()
    
    # Precompute for each t-subset, the list of k-subset indices that contain it
    tsub_to_ksub_indices = {}
    for ti, ts in enumerate(all_tsubs):
        rows, _ = cov_matrix_T.indices[ti], None
        # Get row ti from cov_matrix_T
        start = cov_matrix_T.indptr[ti]
        end = cov_matrix_T.indptr[ti + 1]
        tsub_to_ksub_indices[ts] = cov_matrix_T.indices[start:end]
    
    def greedy_global(seed):
        """Vectorized global greedy: pick the k-subset covering most uncovered t-subsets."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        blocks = []
        used_ksubs = set()
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Compute gain for each unused k-subset
            # gain[i] = number of uncovered t-subsets covered by ksub i
            gains = cov_matrix.dot(uncovered)
            
            # Mask out already-used ksubs
            for b in blocks:
                ki = ksub_to_idx[tuple(b)]
                gains[ki] = -1
            
            best_ki = int(np.argmax(gains))
            best_gain = gains[best_ki]
            
            if best_gain <= 0:
                break
            
            best_block = list(all_ksubs[best_ki])
            blocks.append(best_block)
            used_ksubs.add(best_ki)
            
            # Update uncovered
            idxs = ksub_to_tsub_indices[tuple(best_block)]
            uncovered[idxs] = 0
        
        return blocks

    def remove_redundant(blocks, max_iter=5):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks
        
        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break
                
            # Count coverage of each t-subset
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            block_indices = []
            for b in blocks:
                idxs = ksub_to_tsub_indices[tuple(b)]
                block_indices.append(idxs)
                cover_count[idxs] += 1
            
            valid_blocks = []
            removed_any = False
            
            for i, b in enumerate(blocks):
                if time.time() > deadline - 2.0:
                    return blocks
                
                idxs = block_indices[i]
                if np.all(cover_count[idxs] > 1):
                    cover_count[idxs] -= 1
                    removed_any = True
                else:
                    valid_blocks.append(b)
            
            if not removed_any:
                break
            blocks = valid_blocks
        
        return blocks

    def validate_covering(blocks):
        """Check if blocks form a valid covering."""
        if not blocks:
            return n_tsubs == 0
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in blocks:
            idxs = ksub_to_tsub_indices[tuple(b)]
            covered[idxs] = True
        return np.all(covered)

    def local_search_polish(blocks, max_iters=200):
        """
        Try to reduce block count by finding swaps that maintain coverage.
        For each block, try replacing it with another block that covers
        all the unique t-subsets it was covering.
        """
        if not blocks:
            return blocks
        
        blocks = blocks[:]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Compute coverage counts
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for b in blocks:
                idxs = ksub_to_tsub_indices[tuple(b)]
                cover_count[idxs] += 1
            
            # Try to remove each block
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                
                b = blocks[i]
                idxs = ksub_to_tsub_indices[tuple(b)]
                
                # Can we simply remove this block?
                if np.all(cover_count[idxs] > 1):
                    cover_count[idxs] -= 1
                    blocks.pop(i)
                    improved = True
                    continue
                
                # Find which t-subsets are uniquely covered by this block
                unique_mask = cover_count == 1
                unique_tsubs = [ts for ts in idxs if unique_mask[ts]]
                
                if not unique_tsubs:
                    # All t-subsets it covers are also covered by others
                    cover_count[idxs] -= 1
                    blocks.pop(i)
                    improved = True
                    continue
                
                # Try to find a replacement block that covers all unique_tsubs
                # and is not already in the set
                existing = set(tuple(b) for b in blocks)
                
                # Find candidate replacements: k-subsets that cover all unique_tsubs
                # Start with ksubs covering the first unique tsub
                first_ut = tuple(all_tsubs[unique_tsubs[0]])
                candidates = tsub_to_ksub_indices[first_ut]
                
                best_replace = None
                best_extra_gain = -1
                
                for ki in candidates:
                    if ki in existing:
                        continue
                    ks = all_ksubs[ki]
                    # Check if this ksub covers all unique tsubs
                    ks_tsubs = set(ksub_to_tsub_indices[ks])
                    if all(t in ks_tsubs for t in unique_tsubs):
                        # Compute how many additional t-subsets this covers
                        extra = 0
                        for ti in ks_tsubs:
                            if cover_count[ti] == 0 or (ti not in set(idxs)):
                                extra += 1
                        if extra > best_extra_gain:
                            best_extra_gain = extra
                            best_replace = ks
                
                if best_replace is not None:
                    # Remove old block's coverage
                    cover_count[idxs] -= 1
                    # Add new block's coverage
                    new_idxs = ksub_to_tsub_indices[tuple(best_replace)]
                    cover_count[new_idxs] += 1
                    blocks[i] = list(best_replace)
                    improved = True
        
            if not improved:
                break
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multi-start vectorized global greedy
    num_starts = 0
    start_time = time.time()
    
    # Reserve time for polish
    polish_time_budget = 20.0
    greedy_deadline = deadline - polish_time_budget
    
    while time.time() < greedy_deadline:
        seed = num_starts
        blocks = greedy_global(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        # Reduce
        reduced_blocks = remove_redundant(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
            
        num_starts += 1
    
    if best_blocks is None:
        best_blocks = greedy_global(0)
        best_blocks = remove_redundant(best_blocks)
        best_len = len(best_blocks)
    
    # Phase 2: Local search polish on best solution
    if time.time() < deadline - 5.0:
        polished = local_search_polish(best_blocks, max_iters=500)
        
        if validate_covering(polished) and len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished[:]
    
    # Phase 3: Try a few more greedy starts if time permits, then polish
    if time.time() < deadline - 10.0:
        for extra_seed in range(num_starts, num_starts + 10):
            if time.time() > deadline - 15.0:
                break
            blocks = greedy_global(extra_seed)
            if not blocks:
                continue
            reduced = remove_redundant(blocks)
            if len(reduced) < best_len:
                best_len = len(reduced)
                best_blocks = reduced[:]
        
        # Final polish
        if time.time() < deadline - 5.0:
            polished = local_search_polish(best_blocks, max_iters=300)
            if validate_covering(polished) and len(polished) < best_len:
                best_len = len(polished)
                best_blocks = polished[:]
    
    return best_blocks
