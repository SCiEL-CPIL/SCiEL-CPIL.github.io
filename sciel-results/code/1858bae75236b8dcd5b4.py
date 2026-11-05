import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We should aim to finish by 140s.
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
    
    def greedy_reduce(noise_top_k=1, seed=None, do_polish=True):
        """
        Runs greedy selection with optional randomization among top-k candidates,
        followed by reduction.
        Returns a list of block indices.
        """
        if seed is not None:
            random.seed(seed)
            
        active = np.ones(num_t_subsets, dtype=np.float64)
        selected_block_indices = []
        used_blocks = set()
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            gains = A.dot(active)
            # Mask out already used blocks
            for b in used_blocks:
                gains[b] = 0
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if noise_top_k > 1:
                # Find indices of the top 'noise_top_k' gains
                # np.argsort returns indices that would sort the array.
                # We want the largest, so we reverse or use argpartition.
                # For small k, argsort is fine.
                # Note: If there are ties, this picks arbitrary top k.
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
                
        # Local Search: Deletion
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
                # Check if any subset is covered ONLY by this block
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
                
        # Polish: Try to replace one block with another that covers the unique subsets
        if do_polish and time.time() < deadline - 10.0:
            # Re-compute cover counts after deletion
            cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
            for b_idx in selected_block_indices:
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] += 1
            
            # Find blocks that cover at least one subset uniquely
            # Try to replace them with blocks that cover those unique subsets
            # and potentially more, without breaking the cover.
            
            # For each block, identify the subsets it uniquely covers
            unique_subsets_per_block = {}
            for b_idx in selected_block_indices:
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                unique = [c for c in covered if cover_counts[c] == 1]
                if unique:
                    unique_subsets_per_block[b_idx] = unique
            
            # For each block with unique subsets, try to find a replacement
            # that covers all unique subsets and is not already in the set
            # and has a "good" property (e.g., covers many other subsets)
            
            # We'll iterate a few times to allow cascading improvements
            for _ in range(3):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                new_selected = list(selected_block_indices)
                
                # Re-compute cover counts for the current state
                cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                for b_idx in new_selected:
                    covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                    cover_counts[covered] += 1
                
                for b_idx in new_selected:
                    if time.time() > deadline - 5.0:
                        break
                    covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                    unique = [c for c in covered if cover_counts[c] == 1]
                    if not unique:
                        continue
                    
                    # Find candidate blocks that cover all unique subsets
                    # We can use the matrix to find blocks that intersect with the unique set
                    # A block covers a unique subset if the subset is in the block.
                    # We need a block that contains ALL unique subsets.
                    
                    # This is expensive to check directly.
                    # Approximation: Find blocks that cover the "hardest" unique subset (one with fewest candidates)
                    # and check if they cover the others.
                    
                    # Let's find the unique subset with the fewest covering blocks (excluding current block)
                    # Actually, we know cover_counts[c] == 1, so only b_idx covers it.
                    # So we need a NEW block that covers c.
                    
                    # Let's pick the first unique subset
                    c0 = unique[0]
                    
                    # Find all blocks that cover c0
                    # A[:, c0] gives blocks covering c0
                    # We need to exclude b_idx
                    candidates = A.getcol(c0).indices
                    candidates = [b for b in candidates if b != b_idx]
                    
                    if not candidates:
                        continue
                    
                    # Score candidates by how many other unique subsets they cover
                    # and how many total new subsets they cover
                    best_candidate = None
                    best_score = -1
                    
                    for cand in candidates:
                        if cand in new_selected:
                            continue
                        cand_covered = A.indices[A.indptr[cand]:A.indptr[cand+1]]
                        # Check if cand covers all unique subsets
                        # This is a set intersection check
                        # unique is a list of indices
                        # cand_covered is a list of indices
                        # We need unique subset to be a subset of cand_covered
                        # Since unique is small (usually 1 or 2), we can check directly
                        covers_all = True
                        for u in unique:
                            if u not in cand_covered: # O(N) check, but unique is small
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score: number of subsets covered by cand that are NOT covered by b_idx
                        # or just total coverage
                        # Let's use total coverage as a heuristic
                        score = len(cand_covered)
                        # Bonus for covering other unique subsets? Already checked.
                        # Penalty if cand covers subsets that are uniquely covered by other blocks?
                        # If we replace b_idx with cand, subsets uniquely covered by b_idx are now covered by cand.
                        # Subsets uniquely covered by other blocks are unaffected.
                        # So we just want a valid cover.
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = cand
                    
                    if best_candidate is not None:
                        # Replace b_idx with best_candidate
                        new_selected[new_selected.index(b_idx)] = best_candidate
                        improved = True
                        # Re-compute cover counts immediately? 
                        # Or just continue loop?
                        # Re-computing is safer.
                        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                        for b in new_selected:
                            covered = A.indices[A.indptr[b]:A.indptr[b+1]]
                            cover_counts[covered] += 1
                        # Break inner loop to restart outer loop with new state?
                        # Or just continue?
                        # Let's break and restart the for loop over blocks
                        break
                
                if not improved:
                    break
                
                selected_block_indices = new_selected
                
                # Final deletion pass after polish
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

        return selected_block_indices

    best_blocks = None
    best_size = float('inf')
    
    # 1. Deterministic run (top_k=1)
    res = greedy_reduce(noise_top_k=1, seed=42)
    if len(res) < best_size:
        best_size = len(res)
        best_blocks = res
        
    # 2. Stochastic runs (top_k=5, 10, etc.)
    # Estimate time for one run
    t_start = time.time()
    res = greedy_reduce(noise_top_k=5, seed=random.randint(0, 10000))
    t_elapsed = time.time() - t_start
    
    if len(res) < best_size:
        best_size = len(res)
        best_blocks = res
        
    # Continue stochastic runs if time permits
    # We want to spend remaining time on these.
    # If a run takes ~10ms, we can do many.
    # If it takes ~10s, we do few.
    current_time = time.time()
    remaining = deadline - current_time
    if t_elapsed > 0:
        # Leave a small buffer for the final conversion and safety
        safe_remaining = remaining - 2.0
        num_additional = int(safe_remaining / t_elapsed)
        # Cap additional runs to avoid infinite loops if t_elapsed is tiny
        if num_additional > 100:
             num_additional = 100
             
        for _ in range(num_additional):
            if time.time() > deadline - 5.0:
                break
            # Vary the top_k and seed
            top_k = random.choice([3, 5, 10, 20])
            seed = random.randint(0, 100000)
            res = greedy_reduce(noise_top_k=top_k, seed=seed)
            if len(res) < best_size:
                best_size = len(res)
                best_blocks = res
                
    if best_blocks is None:
        # Fallback: should not happen, but ensure validity
        best_blocks = greedy_reduce(noise_top_k=1, seed=1)
        
    blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return blocks
