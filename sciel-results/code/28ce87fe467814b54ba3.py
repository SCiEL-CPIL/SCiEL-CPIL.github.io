import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute covered t-subsets for each block
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Precompute for each subset, which blocks cover it
    # This is the transpose of A
    A_T = A.T.tocsr()
    
    def greedy_reduce(noise_top_k=1, seed=None, do_polish=True):
        if seed is not None:
            random.seed(seed)
            
        active = np.ones(num_t_subsets, dtype=np.float64)
        selected_block_indices = []
        used_blocks = set()
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            gains = A.dot(active)
            for b in used_blocks:
                gains[b] = 0
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if noise_top_k > 1:
                top_indices = np.argsort(gains)[-noise_top_k:]
                b_idx = random.choice(top_indices)
            else:
                b_idx = np.argmax(gains)
                
            selected_block_indices.append(b_idx)
            used_blocks.add(b_idx)
            
            covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            active[covered_indices] = 0
            
            if np.all(active == 0):
                break
                
        # Local Search
        if do_polish and time.time() < deadline - 10.0:
            # Deletion
            cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
            for b_idx in selected_block_indices:
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] += 1
                
            changed = True
            while changed:
                changed = False
                to_remove = []
                for i, b_idx in enumerate(selected_block_indices):
                    covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                    redundant = True
                    for c in covered:
                        if cover_counts[c] == 1:
                            redundant = False
                            break
                    if redundant:
                        to_remove.append(i)
                        
                if to_remove:
                    for i in to_remove:
                        b_idx = selected_block_indices[i]
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_counts[covered] -= 1
                    remove_set = set(to_remove)
                    selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
                    changed = True
                    
            # 1-for-1 Swap
            if time.time() < deadline - 5.0:
                improved = True
                while improved:
                    improved = False
                    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in selected_block_indices:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_counts[covered] += 1
                    
                    for i, b_idx in enumerate(selected_block_indices):
                        if time.time() > deadline - 2.0:
                            break
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        unique = [c for c in covered if cover_counts[c] == 1]
                        if not unique:
                            continue
                        
                        # Find best replacement
                        best_cand = -1
                        best_score = -1
                        # Get candidates covering the first unique subset
                        c0 = unique[0]
                        candidates = A_T.indices[A_T.indptr[c0]:A_T.indptr[c0+1]]
                        
                        for cand in candidates:
                            if cand == b_idx or cand in used_blocks:
                                continue
                            cand_covered = A.indices[A.indptr[cand]:A.indptr[cand+1]]
                            # Check if covers all unique
                            ok = True
                            for u in unique:
                                # Check if u is in cand_covered
                                # Since cand_covered is sorted, we can binary search or just check
                                # But unique is small.
                                if u not in cand_covered:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            
                            # Score: how many other subsets does it cover?
                            # Or just count
                            score = len(cand_covered)
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                                
                        if best_cand != -1:
                            # Replace
                            used_blocks.remove(b_idx)
                            used_blocks.add(best_cand)
                            selected_block_indices[i] = best_cand
                            improved = True
                            # Re-compute counts
                            cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                            for b in selected_block_indices:
                                covered = A.indices[A.indptr[b]:A.indptr[b+1]]
                                cover_counts[covered] += 1
                            
                            # Try deletion again
                            changed2 = True
                            while changed2:
                                changed2 = False
                                to_remove = []
                                for j, b in enumerate(selected_block_indices):
                                    covered = A.indices[A.indptr[b]:A.indptr[b+1]]
                                    redundant = True
                                    for c in covered:
                                        if cover_counts[c] == 1:
                                            redundant = False
                                            break
                                    if redundant:
                                        to_remove.append(j)
                                if to_remove:
                                    for j in to_remove:
                                        b = selected_block_indices[j]
                                        covered = A.indices[A.indptr[b]:A.indptr[b+1]]
                                        cover_counts[covered] -= 1
                                    remove_set = set(to_remove)
                                    selected_block_indices = [b for j, b in enumerate(selected_block_indices) if j not in remove_set]
                                    changed2 = True
                                    improved = True # Trigger another pass? Or just break?
                            break # Restart the for loop?
                            
            # 2-for-1 Swap
            if time.time() < deadline - 5.0:
                improved = True
                while improved:
                    improved = False
                    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in selected_block_indices:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_counts[covered] += 1
                    
                    # Try pairs
                    n = len(selected_block_indices)
                    for i in range(n):
                        if time.time() > deadline - 2.0:
                            break
                        b1 = selected_block_indices[i]
                        cov1 = set(A.indices[A.indptr[b1]:A.indptr[b1+1]])
                        
                        for j in range(i+1, n):
                            b2 = selected_block_indices[j]
                            cov2 = set(A.indices[A.indptr[b2]:A.indptr[b2+1]])
                            
                            # Find subsets covered ONLY by b1 or b2 (and not others)
                            # Actually, we need subsets that will be UNCOVERED if we remove b1 and b2.
                            # These are subsets s such that cover_counts[s] == 1 and s in b1, or cover_counts[s] == 1 and s in b2, or cover_counts[s] == 2 and s in b1 and s in b2.
                            
                            lost = set()
                            for s in cov1:
                                if cover_counts[s] == 1:
                                    lost.add(s)
                            for s in cov2:
                                if cover_counts[s] == 1:
                                    lost.add(s)
                            # If a subset is in both, and count is 2, it is lost.
                            # The above logic: if s in cov1 and count 1, it's lost. If s in cov2 and count 1, it's lost.
                            # What if s in cov1 and cov2, and count 2?
                            # Then for s in cov1, count is 2, so not added.
                            # For s in cov2, count is 2, so not added.
                            # So we need to handle the count==2 case.
                            for s in cov1:
                                if s in cov2 and cover_counts[s] == 2:
                                    lost.add(s)
                            
                            if not lost:
                                # We can just remove one? No, if lost is empty, it means removing both doesn't uncover anything?
                                # That would mean they are redundant. Deletion pass should have caught single redundancies.
                                # If removing both is safe, we should remove one first.
                                # So this case implies we can remove one of them.
                                # Let's just skip if lost is empty, deletion will handle it.
                                continue
                            
                            # Find a block covering all 'lost'
                            # Intersection of blocks covering each s in lost
                            # Start with all blocks
                            # candidates = set(range(num_blocks))
                            # For each s in lost, candidates &= blocks_covering[s]
                            
                            # Optimize: Pick the s in lost with fewest covering blocks
                            min_s = min(lost, key=lambda s: A_T.indptr[s+1] - A_T.indptr[s])
                            candidates = set(A_T.indices[A_T.indptr[min_s]:A_T.indptr[min_s+1]])
                            
                            for s in lost:
                                if s == min_s:
                                    continue
                                cand_list = A_T.indices[A_T.indptr[s]:A_T.indptr[s+1]]
                                candidates.intersection_update(cand_list)
                                if not candidates:
                                    break
                            
                            # Filter out used blocks
                            candidates -= used_blocks
                            
                            if candidates:
                                # Pick one (e.g., first)
                                new_b = next(iter(candidates))
                                # Replace b1, b2 with new_b
                                used_blocks.remove(b1)
                                used_blocks.remove(b2)
                                used_blocks.add(new_b)
                                
                                # Update list
                                # Remove b1, b2, add new_b
                                # Indices might change, so rebuild list
                                new_list = []
                                for b in selected_block_indices:
                                    if b != b1 and b != b2:
                                        new_list.append(b)
                                new_list.append(new_b)
                                selected_block_indices = new_list
                                improved = True
                                break # Restart pair loop
                        if improved:
                            break
                            
        return selected_block_indices

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple times
    num_runs = 5
    for i in range(num_runs):
        if time.time() > deadline - 20.0:
            break
        seed = i * 100 + 42
        res = greedy_reduce(noise_top_k=5, seed=seed, do_polish=True)
        if len(res) < best_size:
            best_size = len(res)
            best_blocks = res
            
    # If time remains, do more runs or refine
    if best_blocks is None:
        best_blocks = greedy_reduce(noise_top_k=1, seed=1, do_polish=True)
        
    blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return blocks
